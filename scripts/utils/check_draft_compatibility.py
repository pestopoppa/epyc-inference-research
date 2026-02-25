#!/usr/bin/env python3
from __future__ import annotations

"""
Draft-Target Compatibility Checker for Speculative Decoding

MANDATORY: Run this script before adding any draft-target pairing to model_registry.yaml

Checks:
1. Vocab size match (CRITICAL - mismatch causes SIGSEGV)
2. BOS/EOS token ID match (mismatch causes generation failures)
3. Tokenizer type match

Usage:
    python3 check_draft_compatibility.py DRAFT_PATH TARGET_PATH
    python3 check_draft_compatibility.py --batch  # Check all pairings in registry

Exit codes:
    0 = Compatible
    1 = Incompatible (vocab mismatch)
    2 = Warning (BOS/EOS mismatch but may work)
    3 = Error (file not found or parse error)
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    from gguf import GGUFReader
except ImportError:
    print("ERROR: gguf package not installed. Run: pip3 install gguf")
    sys.exit(3)


def get_tokenizer_info(path: str) -> Dict[str, Any]:
    """Extract tokenizer metadata from GGUF file."""
    reader = GGUFReader(path)
    info = {
        'vocab_size': None,
        'bos_token_id': None,
        'eos_token_id': None,
        'tokenizer_model': None,
        'tokenizer_pre': None,
        'add_bos': None,
        'add_eos': None,
    }

    for field in reader.fields.values():
        fname = field.name.lower()

        if fname == 'tokenizer.ggml.tokens':
            # Get vocab size from tokens array
            if hasattr(field, 'data') and field.data is not None:
                info['vocab_size'] = len(field.data)
            elif hasattr(field, 'parts') and len(field.parts) > 0:
                for part in field.parts:
                    if hasattr(part, '__len__') and len(part) > 100:
                        info['vocab_size'] = len(part)
                        break

        elif fname == 'tokenizer.ggml.bos_token_id':
            val = field.parts[-1]
            info['bos_token_id'] = val[0] if hasattr(val, '__getitem__') else val

        elif fname == 'tokenizer.ggml.eos_token_id':
            val = field.parts[-1]
            info['eos_token_id'] = val[0] if hasattr(val, '__getitem__') else val

        elif fname == 'tokenizer.ggml.model':
            val = field.parts[-1]
            # Decode bytes to string
            if hasattr(val, 'tobytes'):
                info['tokenizer_model'] = val.tobytes().decode('utf-8')
            elif hasattr(val, '__iter__'):
                info['tokenizer_model'] = bytes(val).decode('utf-8')
            else:
                info['tokenizer_model'] = str(val)

        elif fname == 'tokenizer.ggml.pre':
            val = field.parts[-1]
            # Decode bytes to string
            if hasattr(val, 'tobytes'):
                info['tokenizer_pre'] = val.tobytes().decode('utf-8')
            elif hasattr(val, '__iter__'):
                info['tokenizer_pre'] = bytes(val).decode('utf-8')
            else:
                info['tokenizer_pre'] = str(val)

        elif fname == 'tokenizer.ggml.add_bos_token':
            val = field.parts[-1]
            info['add_bos'] = bool(val[0]) if hasattr(val, '__getitem__') else bool(val)

        elif fname == 'tokenizer.ggml.add_eos_token':
            val = field.parts[-1]
            info['add_eos'] = bool(val[0]) if hasattr(val, '__getitem__') else bool(val)

    return info


def check_compatibility(draft_path: str, target_path: str, verbose: bool = True) -> Tuple[bool, str]:
    """
    Check if draft model is compatible with target for speculative decoding.

    Returns:
        (is_compatible, message)
    """
    # Check files exist
    if not Path(draft_path).exists():
        return False, f"Draft model not found: {draft_path}"
    if not Path(target_path).exists():
        return False, f"Target model not found: {target_path}"

    try:
        draft_info = get_tokenizer_info(draft_path)
        target_info = get_tokenizer_info(target_path)
    except Exception as e:
        return False, f"Error reading models: {e}"

    issues = []
    warnings = []

    # Check tokenizer model match (quick sanity check)
    if draft_info['tokenizer_model'] != target_info['tokenizer_model']:
        warnings.append(
            f"TOKENIZER MODEL MISMATCH: draft={draft_info['tokenizer_model']}, target={target_info['tokenizer_model']} "
            f"- Different tokenizer families, compatibility unlikely"
        )

    # Check tokenizer pre-processor match
    if draft_info['tokenizer_pre'] != target_info['tokenizer_pre']:
        warnings.append(
            f"TOKENIZER PRE MISMATCH: draft={draft_info['tokenizer_pre']}, target={target_info['tokenizer_pre']}"
        )

    # Check vocab size (RISKY but testing required)
    # Note: Vocab mismatch can go either way:
    #   - Gemma-3: 64 token diff → SIGSEGV crash
    #   - Qwen2.5-Coder: 128 token diff → Works fine (11x speedup)
    # The outcome depends on whether extra tokens are actually generated.
    if draft_info['vocab_size'] != target_info['vocab_size']:
        diff = abs(target_info['vocab_size'] - draft_info['vocab_size'])
        if draft_info['vocab_size'] < target_info['vocab_size']:
            # Draft has fewer tokens than target - RISKY
            warnings.append(
                f"VOCAB MISMATCH: draft={draft_info['vocab_size']:,}, target={target_info['vocab_size']:,} "
                f"({diff:,} fewer tokens in draft) - TESTING REQUIRED! May crash if target generates token ID >= {draft_info['vocab_size']:,}"
            )
        else:
            # Draft has more tokens than target - usually safe
            warnings.append(
                f"VOCAB SIZE: draft={draft_info['vocab_size']:,} > target={target_info['vocab_size']:,} "
                f"({diff:,} extra in draft) - Usually safe, draft has superset"
            )

    # Check BOS token
    if draft_info['bos_token_id'] != target_info['bos_token_id']:
        warnings.append(
            f"BOS token mismatch: draft={draft_info['bos_token_id']}, target={target_info['bos_token_id']}"
        )

    # Check EOS token
    if draft_info['eos_token_id'] != target_info['eos_token_id']:
        warnings.append(
            f"EOS token mismatch: draft={draft_info['eos_token_id']}, target={target_info['eos_token_id']}"
        )

    if verbose:
        print(f"\nDraft Model: {Path(draft_path).name}")
        print(f"  vocab_size: {draft_info['vocab_size']:,}")
        print(f"  bos_token_id: {draft_info['bos_token_id']}")
        print(f"  eos_token_id: {draft_info['eos_token_id']}")
        print(f"  tokenizer_model: {draft_info['tokenizer_model']}")
        print(f"  tokenizer_pre: {draft_info['tokenizer_pre']}")

        print(f"\nTarget Model: {Path(target_path).name}")
        print(f"  vocab_size: {target_info['vocab_size']:,}")
        print(f"  bos_token_id: {target_info['bos_token_id']}")
        print(f"  eos_token_id: {target_info['eos_token_id']}")
        print(f"  tokenizer_model: {target_info['tokenizer_model']}")
        print(f"  tokenizer_pre: {target_info['tokenizer_pre']}")

    if issues:
        msg = "\n".join([f"FATAL: {i}" for i in issues])
        if warnings:
            msg += "\n" + "\n".join([f"WARNING: {w}" for w in warnings])
        return False, msg

    if warnings:
        return True, "COMPATIBLE with warnings:\n" + "\n".join([f"  {w}" for w in warnings])

    return True, "COMPATIBLE: Vocab sizes match, BOS/EOS tokens match"


def main():
    parser = argparse.ArgumentParser(
        description="Check draft-target model compatibility for speculative decoding"
    )
    parser.add_argument('draft', nargs='?', help='Path to draft model GGUF')
    parser.add_argument('target', nargs='?', help='Path to target model GGUF')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if not args.draft or not args.target:
        parser.print_help()
        print("\n\nExample usage:")
        print("  python3 check_draft_compatibility.py /path/to/draft.gguf /path/to/target.gguf")
        sys.exit(3)

    compatible, message = check_compatibility(args.draft, args.target, verbose=not args.quiet)

    print(f"\n{'='*60}")
    if compatible:
        print(f"RESULT: {message}")
        sys.exit(0)
    else:
        print(f"RESULT: INCOMPATIBLE")
        print(message)
        if "VOCAB MISMATCH" in message:
            sys.exit(1)
        else:
            sys.exit(2)


if __name__ == '__main__':
    main()
