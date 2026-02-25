#!/usr/bin/env python3
from __future__ import annotations

"""
Long Context Generator for Benchmark Suite

Generates realistic long context content for the long_context benchmark suite.
Each context type has specific generators that produce appropriate filler content.

Context types:
- technical_docs: Technical documentation with sections
- meeting_notes: Meeting notes with decisions and action items
- structured_doc: Multi-section document with numbered sections
- code_files: Python/JS code files with comments
- python_project: Multi-file Python project structure
- legal_doc: Legal/contract style document
- server_logs: Timestamped server log entries
- business_docs: Quarterly reports and analysis
- full_project: Complete project codebase
"""

import random
import string
from typing import Optional


# Constants for token estimation (conservative - 1 token ~ 4 chars)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    return len(text) // CHARS_PER_TOKEN


def generate_to_tokens(generator_func, target_tokens: int, **kwargs) -> str:
    """Generate content until target token count is reached."""
    content = []
    current_tokens = 0

    while current_tokens < target_tokens:
        chunk = generator_func(**kwargs)
        content.append(chunk)
        current_tokens = estimate_tokens("\n\n".join(content))

    return "\n\n".join(content)


# ============== TECHNICAL DOCUMENTATION ==============

TECH_DOC_TOPICS = [
    ("Configuration", ["environment variables", "config files", "defaults", "overrides"]),
    ("API Reference", ["endpoints", "request format", "response codes", "rate limits"]),
    ("Authentication", ["tokens", "OAuth", "sessions", "permissions"]),
    ("Deployment", ["containers", "scaling", "health checks", "monitoring"]),
    ("Database", ["connections", "migrations", "transactions", "indexes"]),
    ("Networking", ["protocols", "load balancing", "timeouts", "retries"]),
    ("Caching", ["TTL", "invalidation", "distributed cache", "memory limits"]),
    ("Logging", ["log levels", "structured logs", "retention", "aggregation"]),
    ("Security", ["encryption", "certificates", "firewalls", "auditing"]),
    ("Performance", ["profiling", "benchmarks", "optimization", "bottlenecks"]),
]

def generate_tech_doc_section() -> str:
    """Generate a technical documentation section."""
    topic, subtopics = random.choice(TECH_DOC_TOPICS)
    lines = [f"## {topic}\n"]

    for subtopic in subtopics:
        lines.append(f"### {subtopic.title()}\n")
        # Generate 3-5 paragraphs per subtopic
        for _ in range(random.randint(3, 5)):
            words = random.randint(40, 80)
            lines.append(_generate_tech_paragraph(subtopic, words))
        lines.append("")

    return "\n".join(lines)


def _generate_tech_paragraph(topic: str, word_count: int) -> str:
    """Generate a technical paragraph about a topic."""
    templates = [
        f"The {topic} system provides robust handling of various edge cases. ",
        f"When configuring {topic}, ensure that all dependencies are properly initialized. ",
        f"For {topic} operations, the default behavior prioritizes reliability over speed. ",
        f"The {topic} component integrates with the core framework through defined interfaces. ",
        f"Administrators should review {topic} settings during initial deployment. ",
    ]

    filler_words = [
        "This configuration enables", "The system automatically handles",
        "Performance metrics indicate", "Best practices recommend",
        "The implementation follows", "Users should be aware that",
        "This feature was designed to", "The architecture supports",
        "Integration testing confirms", "Documentation specifies",
    ]

    result = random.choice(templates)
    while len(result.split()) < word_count:
        result += f"{random.choice(filler_words)} {_random_tech_phrase()}. "

    return result


def _random_tech_phrase() -> str:
    """Generate a random technical phrase."""
    subjects = ["the service", "each instance", "the controller", "every request", "the handler"]
    verbs = ["processes", "validates", "transforms", "routes", "logs"]
    objects = ["incoming data", "configuration options", "user credentials", "API responses", "system events"]
    return f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}"


# ============== MEETING NOTES ==============

MEETING_TOPICS = [
    "Q4 Planning Session",
    "Engineering Sprint Review",
    "Product Roadmap Discussion",
    "Infrastructure Review",
    "Security Audit Debrief",
    "Customer Feedback Analysis",
    "Budget Allocation Meeting",
    "Team Retrospective",
]

NAMES = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"]

def generate_meeting_notes_section() -> str:
    """Generate meeting notes section with decisions and action items."""
    topic = random.choice(MEETING_TOPICS)
    attendees = random.sample(NAMES, random.randint(4, 7))

    lines = [
        f"## Meeting: {topic}",
        f"Attendees: {', '.join(attendees)}",
        f"Date: 2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "",
        "### Discussion Points",
        ""
    ]

    # Generate discussion items
    for i in range(random.randint(3, 6)):
        lines.append(f"{i+1}. {_generate_discussion_point()}")
        lines.append("")

    # Decisions made
    lines.append("### Decisions Made")
    for i in range(random.randint(2, 4)):
        lines.append(f"- **Decision {i+1}:** {_generate_decision()}")
    lines.append("")

    # Action items
    lines.append("### Action Items")
    for _ in range(random.randint(3, 5)):
        assignee = random.choice(attendees)
        lines.append(f"- [ ] {_generate_action_item()} - Assigned to: {assignee}")

    return "\n".join(lines)


def _generate_discussion_point() -> str:
    topics = [
        "Reviewed the current implementation status of the authentication module",
        "Discussed timeline for the database migration project",
        "Analyzed user feedback regarding the new dashboard features",
        "Evaluated vendor proposals for the infrastructure upgrade",
        "Examined security audit findings and remediation steps",
        "Considered resource allocation for the upcoming quarter",
    ]
    return random.choice(topics) + ". " + _random_tech_phrase().capitalize() + "."


def _generate_decision() -> str:
    decisions = [
        "Proceed with the proposed architecture changes",
        "Postpone the release until all critical bugs are resolved",
        "Allocate additional resources to the testing team",
        "Adopt the new logging framework starting next sprint",
        "Schedule a follow-up meeting to review progress",
        "Implement the suggested security improvements",
    ]
    return random.choice(decisions)


def _generate_action_item() -> str:
    items = [
        "Complete the API documentation review",
        "Set up monitoring dashboards for new services",
        "Create test cases for edge conditions",
        "Update the deployment runbook",
        "Investigate performance bottlenecks",
        "Draft the technical specification document",
        "Coordinate with DevOps on infrastructure changes",
        "Review and approve pending pull requests",
    ]
    return random.choice(items)


# ============== CODE FILES ==============

def generate_code_file_section() -> str:
    """Generate a realistic code file."""
    file_types = [
        ("python", ".py", _generate_python_code),
        ("javascript", ".js", _generate_javascript_code),
        ("config", ".json", _generate_json_config),
    ]

    lang, ext, generator = random.choice(file_types)
    filename = f"{_random_module_name()}{ext}"
    code = generator()

    return f"```{lang}\n# File: {filename}\n{code}\n```"


def _random_module_name() -> str:
    prefixes = ["data", "user", "api", "auth", "cache", "db", "util", "core", "service"]
    suffixes = ["handler", "manager", "processor", "controller", "helper", "utils", "service"]
    return f"{random.choice(prefixes)}_{random.choice(suffixes)}"


def _generate_python_code() -> str:
    """Generate Python code."""
    class_name = ''.join(word.title() for word in _random_module_name().split('_'))

    code = f'''"""
{class_name} module for handling {random.choice(['data processing', 'API requests', 'user management', 'cache operations'])}.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class {class_name}:
    """Handles {random.choice(['incoming requests', 'data transformations', 'user operations', 'cache management'])}."""

    def __init__(self, config: Dict):
        self.config = config
        self._initialized = False
        self._cache = {{}}
        logger.info(f"{{self.__class__.__name__}} initialized")

    def process(self, data: Dict) -> Optional[Dict]:
        """Process incoming data and return result."""
        if not self._initialized:
            self._initialize()

        try:
            result = self._transform(data)
            self._cache[data.get('id')] = result
            return result
        except Exception as e:
            logger.error(f"Processing failed: {{e}}")
            return None

    def _initialize(self) -> None:
        """Initialize internal state."""
        # Setup code here
        self._initialized = True

    def _transform(self, data: Dict) -> Dict:
        """Transform data according to config rules."""
        return {{"processed": True, "original": data}}

    def get_stats(self) -> Dict:
        """Return current statistics."""
        return {{
            "cache_size": len(self._cache),
            "initialized": self._initialized,
        }}
'''
    return code


def _generate_javascript_code() -> str:
    """Generate JavaScript code."""
    module_name = _random_module_name()

    code = f'''/**
 * {module_name} module
 * Handles {random.choice(['API communication', 'state management', 'data validation', 'UI updates'])}
 */

const {{ EventEmitter }} = require('events');

class {module_name.title().replace('_', '')} extends EventEmitter {{
    constructor(options = {{}}) {{
        super();
        this.options = options;
        this.cache = new Map();
        this.initialized = false;
    }}

    async initialize() {{
        if (this.initialized) return;

        // Setup logic
        this.initialized = true;
        this.emit('ready');
    }}

    async process(data) {{
        if (!this.initialized) {{
            await this.initialize();
        }}

        try {{
            const result = this._transform(data);
            this.cache.set(data.id, result);
            return result;
        }} catch (error) {{
            console.error(`Processing failed: ${{error.message}}`);
            throw error;
        }}
    }}

    _transform(data) {{
        return {{ ...data, processed: true, timestamp: Date.now() }};
    }}

    getStats() {{
        return {{
            cacheSize: this.cache.size,
            initialized: this.initialized
        }};
    }}
}}

module.exports = {module_name.title().replace('_', '')};
'''
    return code


def _generate_json_config() -> str:
    """Generate JSON configuration."""
    config = {
        "version": "1.0.0",
        "environment": random.choice(["development", "staging", "production"]),
        "server": {
            "host": "0.0.0.0",
            "port": random.randint(3000, 9000),
            "timeout": random.randint(30, 120),
        },
        "database": {
            "host": "db.internal",
            "port": 5432,
            "pool_size": random.randint(5, 20),
        },
        "cache": {
            "enabled": True,
            "ttl": random.randint(300, 3600),
        },
        "logging": {
            "level": random.choice(["debug", "info", "warn"]),
            "format": "json",
        },
    }

    import json
    return json.dumps(config, indent=2)


# ============== SERVER LOGS ==============

LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]
LOG_SERVICES = ["api-server", "auth-service", "db-proxy", "cache-manager", "worker-01", "worker-02"]

def generate_server_log_section() -> str:
    """Generate server log entries."""
    lines = []

    # Generate a batch of logs for a time period
    base_hour = random.randint(0, 23)
    base_minute = random.randint(0, 50)

    for i in range(random.randint(30, 50)):
        minute = base_minute + (i // 10)
        second = random.randint(0, 59)
        timestamp = f"2024-03-15 {base_hour:02d}:{minute:02d}:{second:02d}"

        level = random.choices(LOG_LEVELS, weights=[10, 60, 20, 10])[0]
        service = random.choice(LOG_SERVICES)
        message = _generate_log_message(level)

        lines.append(f"[{timestamp}] {level:5} [{service}] {message}")

    return "\n".join(lines)


def _generate_log_message(level: str) -> str:
    """Generate a log message appropriate for the level."""
    if level == "DEBUG":
        messages = [
            "Processing request batch #{}".format(random.randint(1000, 9999)),
            "Cache lookup for key: user_{}".format(random.randint(100, 999)),
            "Query execution time: {}ms".format(random.randint(1, 50)),
            "Connection pool status: {}/20 active".format(random.randint(1, 20)),
        ]
    elif level == "INFO":
        messages = [
            "Request completed successfully (200 OK)",
            "User authenticated: user_{}".format(random.randint(100, 999)),
            "Scheduled job completed: daily_cleanup",
            "New connection established from 10.0.{}.{}".format(random.randint(0, 255), random.randint(0, 255)),
            "Configuration reloaded",
        ]
    elif level == "WARN":
        messages = [
            "High memory usage detected: {}%".format(random.randint(75, 95)),
            "Slow query detected ({}ms)".format(random.randint(500, 2000)),
            "Rate limit approaching for client_{}".format(random.randint(100, 999)),
            "Retry attempt {} for external API call".format(random.randint(1, 3)),
        ]
    else:  # ERROR
        messages = [
            "Connection refused to database",
            "Authentication failed for user_{}".format(random.randint(100, 999)),
            "Request timeout after 30s",
            "Invalid JSON payload received",
            "Service unavailable: external-api",
        ]

    return random.choice(messages)


# ============== LEGAL DOCUMENT ==============

def generate_legal_doc_section() -> str:
    """Generate legal/contract style content."""
    section_num = random.randint(1, 20)

    lines = [
        f"## Section {section_num}: {_random_legal_title()}",
        "",
    ]

    # Add subsections
    for i in range(random.randint(3, 6)):
        lines.append(f"### {section_num}.{i+1} {_random_legal_subtitle()}")
        lines.append("")

        # Legal paragraphs
        for _ in range(random.randint(2, 4)):
            lines.append(_generate_legal_paragraph())
            lines.append("")

    return "\n".join(lines)


def _random_legal_title() -> str:
    titles = [
        "Representations and Warranties",
        "Indemnification",
        "Limitation of Liability",
        "Confidentiality Obligations",
        "Termination Rights",
        "Intellectual Property",
        "Dispute Resolution",
        "Force Majeure",
        "Assignment and Transfer",
        "Compliance Requirements",
    ]
    return random.choice(titles)


def _random_legal_subtitle() -> str:
    subtitles = [
        "General Provisions",
        "Scope of Application",
        "Exceptions and Limitations",
        "Notice Requirements",
        "Remedies Available",
        "Survival of Obligations",
    ]
    return random.choice(subtitles)


def _generate_legal_paragraph() -> str:
    """Generate a legal-style paragraph."""
    templates = [
        "Notwithstanding any provision to the contrary herein, the parties agree that",
        "Subject to the terms and conditions set forth in this Agreement,",
        "In the event of a breach of the foregoing obligations,",
        "The parties hereby acknowledge and agree that",
        "Without limiting the generality of the foregoing,",
    ]

    continuations = [
        "all rights and remedies shall remain available to the non-breaching party.",
        "the obligations set forth herein shall survive termination of this Agreement.",
        "neither party shall be liable for any indirect, incidental, or consequential damages.",
        "the prevailing party shall be entitled to recover reasonable attorney's fees.",
        "any modification hereto must be in writing and signed by both parties.",
    ]

    # Add some monetary amounts and dates
    amounts = [
        f"${random.randint(1, 100) * 10000:,}",
        f"${random.randint(1, 50) * 100000:,}",
    ]
    dates = [
        f"January {random.randint(1, 28)}, 2024",
        f"March {random.randint(1, 31)}, 2024",
        f"the {random.randint(1, 30)}th day of {random.choice(['April', 'May', 'June'])}, 2024",
    ]

    para = f"{random.choice(templates)} {random.choice(continuations)} "

    if random.random() > 0.5:
        para += f"The total liability shall not exceed {random.choice(amounts)}. "
    if random.random() > 0.5:
        para += f"This provision shall become effective as of {random.choice(dates)}. "

    return para


# ============== BUSINESS DOCUMENTS ==============

def generate_business_doc_section() -> str:
    """Generate business report content."""
    doc_types = [
        ("Q1 Financial Report", _generate_quarterly_report),
        ("Market Analysis", _generate_market_analysis),
        ("Competitor Assessment", _generate_competitor_assessment),
    ]

    title, generator = random.choice(doc_types)
    return f"## {title}\n\n{generator()}"


def _generate_quarterly_report() -> str:
    """Generate quarterly financial report content."""
    lines = []

    lines.append("### Executive Summary\n")
    lines.append(f"Revenue for the quarter reached ${random.randint(10, 100)}M, "
                 f"representing a {random.randint(-10, 30)}% change from the previous period. "
                 f"Operating margin improved to {random.randint(10, 40)}%.\n")

    lines.append("### Key Metrics\n")
    lines.append(f"- Total Revenue: ${random.randint(10, 100)}M")
    lines.append(f"- Gross Profit: ${random.randint(5, 50)}M")
    lines.append(f"- Operating Expenses: ${random.randint(3, 20)}M")
    lines.append(f"- Net Income: ${random.randint(1, 30)}M")
    lines.append(f"- Customer Count: {random.randint(1000, 50000):,}\n")

    lines.append("### Analysis\n")
    lines.append(_generate_business_paragraph())

    return "\n".join(lines)


def _generate_market_analysis() -> str:
    """Generate market analysis content."""
    lines = []

    lines.append("### Market Overview\n")
    lines.append(f"The total addressable market is estimated at ${random.randint(1, 100)}B, "
                 f"with a projected CAGR of {random.randint(5, 25)}% over the next five years.\n")

    lines.append("### Key Trends\n")
    trends = [
        "Digital transformation acceleration",
        "Increased focus on sustainability",
        "Remote work normalization",
        "AI/ML adoption growth",
        "Cybersecurity priority increase",
    ]
    for trend in random.sample(trends, 3):
        lines.append(f"- {trend}")
    lines.append("")

    lines.append("### Market Segments\n")
    lines.append(_generate_business_paragraph())

    return "\n".join(lines)


def _generate_competitor_assessment() -> str:
    """Generate competitor analysis content."""
    lines = []

    competitors = ["TechCorp", "DataSystems Inc", "CloudFirst", "InnovateTech", "GlobalSoft"]

    lines.append("### Competitive Landscape\n")

    for comp in random.sample(competitors, 3):
        lines.append(f"#### {comp}")
        lines.append(f"- Market share: {random.randint(5, 30)}%")
        lines.append(f"- Key strength: {random.choice(['Enterprise focus', 'Product innovation', 'Price leadership', 'Customer service'])}")
        lines.append(f"- Weakness: {random.choice(['Limited scalability', 'Legacy technology', 'Regional focus', 'High costs'])}")
        lines.append("")

    return "\n".join(lines)


def _generate_business_paragraph() -> str:
    """Generate a business-style paragraph."""
    openings = [
        "Analysis indicates that market conditions remain favorable",
        "Strategic initiatives have yielded positive results",
        "Customer feedback suggests strong product-market fit",
        "Operational improvements continue to drive efficiency",
    ]

    continuations = [
        "with continued growth expected in key segments.",
        "although challenges remain in emerging markets.",
        "supported by strong demand in enterprise sectors.",
        "enabling sustained competitive advantage.",
    ]

    return f"{random.choice(openings)}, {random.choice(continuations)}"


# ============== MAIN CONTEXT GENERATOR ==============

GENERATORS = {
    "technical_docs": generate_tech_doc_section,
    "meeting_notes": generate_meeting_notes_section,
    "structured_doc": generate_legal_doc_section,  # Using legal as structured
    "code_files": generate_code_file_section,
    "python_project": generate_code_file_section,  # Same generator, multiple files
    "legal_doc": generate_legal_doc_section,
    "server_logs": generate_server_log_section,
    "business_docs": generate_business_doc_section,
    "full_project": generate_code_file_section,  # Same generator, more files
}


def generate_context(
    context_type: str,
    target_tokens: int,
    needle: Optional[str] = None,
    needle_position: str = "middle",
) -> str:
    """Generate long context content with optional needle insertion.

    Args:
        context_type: Type of context to generate (e.g., 'technical_docs', 'code_files')
        target_tokens: Approximate number of tokens to generate
        needle: Optional string to insert in the context
        needle_position: Where to insert needle ('early', 'middle', 'deep', 'very_deep')

    Returns:
        Generated context string with needle inserted if provided.
    """
    generator = GENERATORS.get(context_type, generate_tech_doc_section)

    # Generate content in chunks
    chunks = []
    current_tokens = 0

    while current_tokens < target_tokens:
        chunk = generator()
        chunks.append(chunk)
        current_tokens = estimate_tokens("\n\n".join(chunks))

    # Insert needle if provided
    if needle:
        position_map = {
            "early": 0.25,
            "middle": 0.5,
            "deep": 0.75,
            "very_deep": 0.85,
        }
        position_ratio = position_map.get(needle_position, 0.5)

        # Find insertion point
        insert_idx = int(len(chunks) * position_ratio)
        insert_idx = max(1, min(insert_idx, len(chunks) - 1))

        # Insert needle as its own chunk
        chunks.insert(insert_idx, needle)

    return "\n\n".join(chunks)


def build_full_prompt(
    context_type: str,
    target_tokens: int,
    question_prompt: str,
    needle: Optional[str] = None,
    needle_position: str = "middle",
) -> str:
    """Build complete prompt with context + question.

    Args:
        context_type: Type of context to generate
        target_tokens: Approximate tokens for context
        question_prompt: The actual question/instruction
        needle: Optional needle to hide in context
        needle_position: Where to place the needle

    Returns:
        Complete prompt: context + separator + question
    """
    context = generate_context(
        context_type=context_type,
        target_tokens=target_tokens,
        needle=needle,
        needle_position=needle_position,
    )

    return f"{context}\n\n---\n\n{question_prompt}"


if __name__ == "__main__":
    print("=== Context Generator Test ===\n")

    # Test each context type
    for ctx_type in GENERATORS:
        print(f"\n--- {ctx_type} ---")
        content = generate_context(ctx_type, target_tokens=500)
        tokens = estimate_tokens(content)
        print(f"Generated ~{tokens} tokens")
        print(content[:500] + "...\n")

    # Test needle insertion
    print("\n--- Needle Test ---")
    context = generate_context(
        context_type="server_logs",
        target_tokens=1000,
        needle="[CRITICAL] SECRET_KEY=abc123xyz",
        needle_position="deep",
    )

    if "SECRET_KEY=abc123xyz" in context:
        pos = context.find("SECRET_KEY=abc123xyz")
        print(f"Needle found at position {pos}/{len(context)} ({pos/len(context)*100:.1f}%)")
    else:
        print("ERROR: Needle not found!")
