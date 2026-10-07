"""File exploration tools for the REPL environment.

Read-only file system inspection: peek, grep, list_dir, file_info.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def _explicit_exploration_root(env, file_path: str) -> tuple[str, str]:
    """Return a validated resolved path and its caller-authorized explicit root.

    Callers must run ``_validate_file_path`` first; that remains the authority for
    task scope, legacy prefixes, and the knowledge-fence recording order.
    """
    from src.repl_environment.task_root import (
        get_task_root, request_scope, resolve_task_path, task_root_active,
    )

    resolved = os.path.realpath(resolve_task_path(file_path))
    scope = request_scope()
    if scope is not None:
        roots = (scope.root, *scope.read_roots)
    elif task_root_active():
        roots = (str(get_task_root()),)
    else:
        # Legacy admission has already checked the configured lexical prefix and
        # knowledge fence. For this single-file read, the resolved file's parent
        # is the narrow explicit root; no second prefix policy is introduced.
        return os.path.dirname(resolved), resolved

    candidates = []
    for value in roots:
        root = os.path.realpath(os.fspath(value)).rstrip(os.sep) or os.sep
        try:
            if os.path.commonpath((root, resolved)) == root:
                candidates.append(root)
        except ValueError:
            continue
    if not candidates:
        raise PermissionError(f"validated path has no matching explicit exploration root: {resolved}")
    return max(candidates, key=len), resolved


class _FileExplorationMixin:
    """Mixin providing read-only file system exploration tools (_peek, _grep, _list_dir, _file_info).

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        context: str — full input context
        artifacts: dict — collected artifacts
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        _grep_hits_buffer: list — grep results buffer for two-stage summarization
        _validate_file_path: Callable[[str], tuple[bool, str | None]] — path validation method
        _research_context: ResearchContext — research context tracker
        _last_research_node: str | None — last research node ID
    """

    def _increment_exploration(self) -> None:
        """Thread-safe increment of exploration call counter."""
        lock = getattr(self, "_state_lock", None)
        if lock:
            with lock:
                self._exploration_calls += 1
        else:
            self._exploration_calls += 1

    def _track_research(self, tool: str, query: str, content: str) -> None:
        """Track a tool invocation in the research context.

        Args:
            tool: Tool name
            query: Tool arguments/query
            content: Tool output content
        """
        if hasattr(self, "_research_context"):
            try:
                node_id = self._research_context.add(
                    tool=tool,
                    query=query,
                    content=content[:8000],
                    parent_id=getattr(self, "_last_research_node", None),
                )
                self._last_research_node = node_id
            except Exception:
                pass  # Silently ignore research tracking failures

    @staticmethod
    def _page_text(text: str, n: int, offset: int) -> str:
        """Slice ``n`` characters of ``text`` starting at ``offset`` (negative = from the end).

        At offset 0 this is exactly the historical ``context[:n]``.
        """
        offset = int(offset)
        if offset < 0:
            offset = max(0, len(text) + offset)
        return text[offset:][:int(n)]

    @staticmethod
    def _read_file_page(path: str, n: int, offset: int) -> str:
        """Read ``n`` characters of a file starting at character ``offset``.

        ``newline=""`` disables newline translation so character offsets match the
        text that was written (exact recall of spilled output, TOC-SP-2). A
        negative offset counts from the end of the file. A negative ``n`` reads to
        the end, as the historical ``f.read(n)`` did.
        """
        n = int(n)
        offset = int(offset)
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            if offset < 0:
                text = f.read()
                start = max(0, len(text) + offset)
                return text[start:] if n < 0 else text[start:start + n]
            remaining = offset
            while remaining > 0:
                chunk = f.read(min(remaining, 1 << 20))
                if not chunk:
                    return ""
                remaining -= len(chunk)
            return f.read(n)

    def _peek(self, n: int = 500, file_path: str | None = None, offset: int = 0) -> str:
        """Return n characters of context or file, starting at ``offset``.

        Args:
            n: Number of characters to return (default 500).
            file_path: Optional file path to read from instead of context.
            offset: Character offset to start from (default 0 = the start; a
                negative value counts back from the end). Together with ``n``
                this pages through a file exactly, e.g. a spilled tool output.

        Returns:
            Up to n characters of the context or file from ``offset``.
        """
        self._increment_exploration()
        try:
            n = int(n)
            offset = int(offset)
        except (TypeError, ValueError):
            return f"[ERROR: peek(n, file_path=None, offset=0) needs integer n/offset, got n={n!r}, offset={offset!r}]"

        if file_path is not None:
            # Read from file
            is_valid, error = self._validate_file_path(file_path)
            if not is_valid:
                return f"[ERROR: {error}]"
            # Resolve to the task-root (mirror file_write_safe) so reads find files where
            # writes + task setup put them. No-op in prod (task-root inactive → realpath).
            try:
                from src.repl_environment.exploration_core import read_file_page

                root, resolved = _explicit_exploration_root(self, file_path)
                result = read_file_page(root, resolved, n, offset, max_bytes=None)
                event = {"n": n, "file_path": file_path}
                if offset:
                    event["offset"] = offset
                self._exploration_log.add_event("peek", event, result)
                self._track_research("peek", f"n={n}, offset={offset}, file={file_path}", result)
                return result
            except FileNotFoundError:
                return f"[ERROR: File not found: {file_path}]"
            except Exception as e:
                logger.debug("peek failed", exc_info=True)
                return f"[ERROR: {type(e).__name__}: {e}]"

        # Read from context
        bundle = getattr(self, "_context_bundle", None)
        if bundle is not None:
            # INF-78 OAB-7: `context` is the request's bundle; count the pull, keep it out
            # of the research tracker (whose summaries render into the root prompt).
            try:
                return bundle.page(n, offset, op="peek")
            except Exception as e:  # ContextPullBudgetExceeded
                return f"[ERROR: {e}]"
        result = self._page_text(self.context, n, offset)
        event = {"n": n}
        if offset:
            event["offset"] = offset
        self._exploration_log.add_event("peek", event, result)
        self._track_research("peek", f"n={n}" + (f", offset={offset}" if offset else ""), result)
        return result

    def _grep(
        self,
        pattern: str,
        file_path: str | None = None,
        context_lines: int = 2,
    ) -> list[str]:
        """Search context or file with regex and return matching lines.

        Also captures hits to _grep_hits_buffer for two-stage summarization.

        Args:
            pattern: Regular expression pattern to search for.
            file_path: Optional file path to search instead of context.
            context_lines: Number of context lines before/after match (default 2).

        Returns:
            List of lines containing matches (capped at max_grep_results).
        """
        self._increment_exploration()
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [f"[REGEX ERROR: {e}]"]

        # Determine source text
        source_name = "context"
        bundle = getattr(self, "_context_bundle", None)
        if file_path is None and bundle is not None:
            # INF-78 OAB-7: grep the request's bundle, counted (see _peek).
            try:
                hits = bundle.grep(pattern, k=self.config.max_grep_results, op="grep_legacy")
            except Exception as e:  # bad regex already handled above; budget exhaustion
                return [f"[ERROR: {e}]"]
            lines = [h["text"] for h in hits]
            if len(hits) >= self.config.max_grep_results:
                lines.append(f"[... truncated at {self.config.max_grep_results} results]")
            return lines
        if file_path is not None:
            is_valid, error = self._validate_file_path(file_path)
            if not is_valid:
                return [f"[ERROR: {error}]"]
            # Resolve to task-root (mirror file_write_safe); no-op in prod.
            try:
                from src.repl_environment.exploration_core import grep_file

                root, resolved = _explicit_exploration_root(self, file_path)
                result = grep_file(
                    root, resolved, pattern, context_lines=context_lines,
                    max_hits=self.config.max_grep_results, max_line_chars=None,
                    max_context_chars=None, max_bytes=None, allow_empty_pattern=True,
                )
                source_text = None
                source_name = file_path
            except FileNotFoundError:
                return [f"[ERROR: File not found: {file_path}]"]
            except Exception as e:
                logger.debug("grep failed", exc_info=True)
                return [f"[ERROR: {type(e).__name__}: {e}]"]
        else:
            source_text = self.context

        if file_path is not None:
            matches = [item.line for item in result.matches]
            match_details = [
                {"line_num": item.line_num, "match": item.line[:500], "context": item.context[:1000]}
                for item in result.matches
            ]
            if match_details and len(matches) >= self.config.max_grep_results:
                matches.append(f"[... truncated at {self.config.max_grep_results} results]")
            if match_details:
                self._grep_hits_buffer.append({
                    "pattern": pattern, "source": source_name,
                    "match_count": len(match_details), "hits": match_details[:20],
                })
            self._exploration_log.add_event(
                "grep", {"pattern": pattern, "file_path": file_path}, matches
            )
            self._track_research(
                "grep", f"pattern={pattern!r}, file={file_path}", "\n".join(matches[:10])
            )
            return matches

        lines = source_text.split("\n")
        matches = []
        match_details = []  # For grep hits buffer

        for i, line in enumerate(lines):
            if regex.search(line):
                matches.append(line)

                # Capture context for grep hits buffer
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context_snippet = "\n".join(lines[start:end])

                match_details.append(
                    {
                        "line_num": i + 1,
                        "match": line[:500],  # Cap line length
                        "context": context_snippet[:1000],  # Cap context
                    }
                )

                if len(matches) >= self.config.max_grep_results:
                    matches.append(f"[... truncated at {self.config.max_grep_results} results]")
                    break

        # Store in grep hits buffer for two-stage pipeline
        if match_details:
            self._grep_hits_buffer.append(
                {
                    "pattern": pattern,
                    "source": source_name,
                    "match_count": len(match_details),
                    "hits": match_details[:20],  # Cap at 20 detailed hits
                }
            )

        self._exploration_log.add_event(
            "grep", {"pattern": pattern, "file_path": file_path}, matches
        )
        self._track_research(
            "grep",
            f"pattern={pattern!r}" + (f", file={file_path}" if file_path else ""),
            "\n".join(matches[:10]),
        )
        return matches

    def _outline_file(self, file_path: str) -> str:
        """Return a bounded heuristic source outline for an admitted file path.

        This adapter is intentionally not registered as a REPL/MCP tool yet; tool
        exposure remains behind the separate catalog and rerun gate.
        """
        self._increment_exploration()
        is_valid, error = self._validate_file_path(file_path)
        if not is_valid:
            return f"[ERROR: {error}]"
        try:
            from src.repl_environment.exploration_core import outline_file

            root, resolved = _explicit_exploration_root(self, file_path)
            result = outline_file(root, resolved)
            self._exploration_log.add_event("outline_file", {"file_path": file_path}, result)
            self._track_research("outline_file", f"file={file_path}", result)
            return result
        except FileNotFoundError:
            return f"[ERROR: File not found: {file_path}]"
        except IsADirectoryError:
            return f"[ERROR: Path is a directory: {file_path}]"
        except Exception as e:
            logger.debug("outline_file failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _list_dir(self, path: str) -> str:
        """List contents of a directory.

        Args:
            path: Absolute path to the directory.

        Returns:
            JSON string with directory contents.
        """
        self._increment_exploration()
        import json
        import os

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        try:
            entries = []
            # List the directory that was VALIDATED: a relative path resolves under the task
            # root when one is active (BEP env / INF-78 request scope); inactive, this is
            # os.path.realpath(path) — the same directory the raw path named.
            from src.repl_environment.task_root import resolve_task_path

            for entry in os.scandir(resolve_task_path(path)):
                entry_info = {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                }
                if entry.is_file():
                    try:
                        entry_info["size"] = entry.stat().st_size
                    except Exception:
                        logger.debug("Failed to stat %s", entry.name, exc_info=True)
                        entry_info["size"] = 0
                entries.append(entry_info)

            # Sort: directories first, then files
            entries.sort(key=lambda x: (x["type"] == "file", x["name"]))

            # Frecency-weighted re-sort within type groups (feature-flagged)
            if os.environ.get("REPL_FRECENCY", "").lower() in ("1", "true", "on"):
                try:
                    from src.repl_environment.file_recency import FrecencyStore

                    _frecency = getattr(self, "_frecency_store", None)
                    if _frecency is None:
                        _frecency = FrecencyStore()
                        self._frecency_store = _frecency
                    full_paths = [os.path.join(path, e["name"]) for e in entries]
                    scores = _frecency.get_scores(full_paths)
                    # Stable sort: within each type group, sort by descending frecency
                    entries.sort(
                        key=lambda x: (
                            x["type"] == "file",
                            -scores.get(os.path.join(path, x["name"]), 0.0),
                        )
                    )
                    _frecency.record_access(path)
                except Exception:
                    logger.debug("Frecency sorting failed", exc_info=True)

            result = {
                "path": path,
                "files": entries[:100],  # Cap at 100 entries
                "total": len(entries),
            }

            self._exploration_log.add_event("list_dir", {"path": path}, result)
            self._track_research("list_dir", f"path={path}", str(result))

            # Use TOON encoding for token efficiency if enabled
            if self.config.use_toon_encoding:
                from src.services.toon_encoder import encode_list_dir

                output = encode_list_dir(path, entries[:100], len(entries))
            else:
                output = json.dumps(result, indent=2)
            return self._maybe_wrap_tool_output(output)

        except FileNotFoundError:
            return f"[ERROR: Directory not found: {path}]"
        except NotADirectoryError:
            return f"[ERROR: Not a directory: {path}]"
        except PermissionError:
            return f"[ERROR: Permission denied: {path}]"
        except Exception as e:
            logger.debug("list_dir failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _file_info(self, path: str) -> str:
        """Get metadata about a file.

        Args:
            path: Absolute path to the file.

        Returns:
            JSON string with file metadata.
        """
        self._increment_exploration()
        import json
        import os
        from datetime import datetime

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        # Resolve to task-root (mirror file_write_safe); no-op in prod.
        from src.repl_environment.task_root import resolve_task_path
        _rp = resolve_task_path(path)
        try:
            stat_info = os.stat(_rp)
            is_dir = os.path.isdir(_rp)
            is_link = os.path.islink(_rp)

            result = {
                "path": path,
                "exists": True,
                "type": "symlink" if is_link else ("dir" if is_dir else "file"),
                "size": stat_info.st_size,
                "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "extension": os.path.splitext(path)[1] if not is_dir else None,
            }

            self._exploration_log.add_event("file_info", {"path": path}, result)
            return json.dumps(result, indent=2)

        except FileNotFoundError:
            return json.dumps({"path": path, "exists": False})
        except Exception as e:
            logger.debug("file_info failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"
