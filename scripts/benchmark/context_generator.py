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

# =============================================================================
# Tier-3 long_context generators (added 2026-05-06)
# =============================================================================
# Pre-2026-05-06: t3_q1_multi_hop_reasoning, t3_q2_contradiction_detection,
# t3_q3_evolving_requirements all referenced context_types that weren't in the
# GENERATORS dict, so generate_context() silently fell back to tech-docs. All
# models correctly refused on the wrong-domain context, so the rubric couldn't
# discriminate quality. The three generators below produce contexts with the
# domain structure each question expects.

def generate_investigation_docs_section() -> str:
    """Financial investigation timeline with cross-document references.

    Supports t3_q1_multi_hop_reasoning: trace transactions A→B→C, identify
    Person D's role, find meeting between D and E. Provides specific entities,
    dates, amounts, and roles spread across multiple sub-documents to require
    multi-hop synthesis.
    """
    base_year = 2024
    date_x_month = random.randint(2, 6)
    date_x_day = random.randint(1, 25)
    date_y_month = date_x_month if date_x_day <= 14 else date_x_month + 1
    date_y_day = (date_x_day + 14) % 28 + 1
    amount_initial = random.randint(2, 9) * 100000
    amount_transfer = int(amount_initial * random.choice([0.85, 0.90, 0.95]))

    person_d = random.choice(NAMES)
    person_e = random.choice([n for n in NAMES if n != person_d])
    company_a = f"Acme Industries LLC"
    company_b = f"Crestline Holdings Inc."
    account_c = f"Pacific Trust Account #C-{random.randint(10000, 99999)}"

    sections = []

    sections.append(f"""## Document 1: Wire Transfer Records — {company_a}

Transaction reference: WT-{random.randint(1000, 9999)}-{base_year}
Date: {base_year}-{date_x_month:02d}-{date_x_day:02d}
Originator: {company_a}
Beneficiary: {company_b}
Amount: ${amount_initial:,}.00 USD
Memo: Consulting services per contract dated {base_year}-{(date_x_month-1):02d}-15
Status: COMPLETED
Compliance flags: None at time of transfer

Notes from compliance review (added {base_year}-{date_x_month:02d}-{date_x_day+3:02d}):
The amount and timing fall within normal parameters for the consulting agreement.
Additional review may be warranted if pattern continues.""")

    sections.append(f"""## Document 2: Banking Activity — {company_b}

Account holder: {company_b}
Statement period: {base_year}-{date_x_month:02d}-01 to {base_year}-{date_x_month:02d}-30

Notable transactions:
- {base_year}-{date_x_month:02d}-{date_x_day:02d} INCOMING WIRE ${amount_initial:,}.00 from {company_a}
- {base_year}-{date_x_month:02d}-{(date_x_day+5):02d} ACH DEBIT $4,250.00 (operating expenses)
- {base_year}-{date_y_month:02d}-{date_y_day:02d} OUTGOING WIRE ${amount_transfer:,}.00 to {account_c}
- {base_year}-{date_y_month:02d}-{(date_y_day+2):02d} ACH DEBIT $1,800.00 (legal fees)

The {base_year}-{date_y_month:02d}-{date_y_day:02d} transfer to {account_c} occurred
approximately two weeks after the inbound wire from {company_a}. Funds breakdown:
${amount_transfer:,} of the ${amount_initial:,} received was forwarded; the
${amount_initial - amount_transfer:,} difference was retained as reserves.""")

    sections.append(f"""## Document 3: Personnel File — {person_d}

Employee ID: EMP-{random.randint(2000, 5999)}
Position: Senior Compliance Officer
Department: Risk & Audit
Start date: {base_year-2}-09-15
Reports to: Chief Risk Officer

Role responsibilities:
- Reviews and approves wire transfers exceeding $250,000 threshold
- Coordinates with external auditors on quarterly reviews
- Maintains the compliance exception log
- Authorizes deviations from standard counterparty protocols

Recent activity ({base_year} Q1-Q2):
{person_d} signed off on 14 wire transfers including the {company_a} → {company_b}
transaction on {base_year}-{date_x_month:02d}-{date_x_day:02d}. Compliance log entry
notes "approved per existing consulting framework" with reference to the standing
{company_b} relationship.""")

    sections.append(f"""## Document 4: Calendar Records — {person_e}

Position: Director of Strategic Partnerships, {company_b}

{base_year}-{(date_x_month-1):02d}-22  09:00–10:30  Internal: Quarterly review prep
{base_year}-{(date_x_month-1):02d}-25  14:00–15:00  PRIVATE MEETING with {person_d} (off-site)
                                                     Location: Westin Hotel, Conference Room B
                                                     Notes: Discussion of upcoming engagement structure
{base_year}-{(date_x_month-1):02d}-28  11:00–12:00  Contract review with legal
{base_year}-{date_x_month:02d}-02  10:00–11:00  Internal: Engagement kickoff
{base_year}-{date_x_month:02d}-{date_x_day:02d}  --:--   [Wire transfer occurred — see Doc 1]
{base_year}-{date_x_month:02d}-{(date_x_day+1):02d}  16:00–17:00  Follow-up call with {person_d}

The {base_year}-{(date_x_month-1):02d}-25 meeting between {person_e} and {person_d}
preceded the {company_a} wire transfer by approximately three weeks. The meeting
was scheduled outside normal business premises and was not recorded in the
official engagement minutes.""")

    return "\n\n".join(sections)


def generate_witness_statements_section() -> str:
    """Eight witness statements about an incident with embedded contradictions.

    Supports t3_q2_contradiction_detection: provides 8 narrative statements
    with deliberate factual conflicts (timing, location, perpetrator features),
    reliability variation (some witnesses with line-of-sight issues, some
    impaired), and enough detail to construct a synthesized timeline.
    """
    incident_date = f"2024-0{random.randint(2,6)}-{random.randint(10,25)}"
    base_hour = random.randint(20, 22)
    locations = ["First and Main", "Second Avenue near the post office",
                 "the corner of Oak Street", "the parking lot behind the bank"]
    base_loc = random.choice(locations)
    alt_loc = random.choice([l for l in locations if l != base_loc])
    suspect_height = random.choice(["5'10\"", "6'1\"", "5'8\""])
    alt_height = random.choice([h for h in ["5'10\"", "6'1\"", "5'8\"", "6'3\""] if h != suspect_height])

    statements = [
        f"""## Witness Statement #1: Margaret Chen (cashier, line of sight: clear, sober)

I was working the register at the time. I heard the noise at exactly {base_hour}:15 PM
on {incident_date}. The man came in through the front door wearing a dark blue jacket
and a black baseball cap. He was tall, about {suspect_height}. He walked directly to my
register without looking around. He didn't say anything for the first few seconds, just
slid a note across the counter. The note said "give me the cash, no alarms." I followed
the protocol we trained for. He left within two minutes, carrying a black backpack.""",

        f"""## Witness Statement #2: Robert Alvarez (customer, line of sight: partial, sober)

I was in the parking lot when it happened. Looking at my phone, the time was {base_hour}:25 PM.
I saw a man walk out the front of the building at a fast pace. He was wearing what looked
like a black hoodie — definitely dark colors. I couldn't see his height clearly because he
was at an angle, but he seemed tall. He went around the side of the building toward the alley.
A few seconds later I heard sirens. I didn't see his face.""",

        f"""## Witness Statement #3: Diane Foster (customer, line of sight: blocked, sober)

I was near the back of the store browsing the magazine section. I never actually saw
the man — there was a display blocking my view of the counter. But I heard him speak.
He had a deep voice and an accent I couldn't place — maybe East Coast? He said something
like "this is a robbery, stay calm." Then I heard the cashier respond. The whole thing
took maybe 90 seconds. I called 911 from where I was hiding.""",

        f"""## Witness Statement #4: James Park (security guard, line of sight: clear, on duty)

I was monitoring the camera feed from the back office. The man entered at {base_hour}:14 PM
on the timestamp. White male, approximately {suspect_height}, wearing a dark jacket — I'd
say navy, not black — and a baseball cap. NO BACKPACK was visible when he entered. He was
carrying a folded piece of paper in his right hand. He left at {base_hour}:17 PM, this time
carrying what appeared to be a black bag. Total time inside: 3 minutes 12 seconds per the
timestamps.""",

        f"""## Witness Statement #5: Elena Vasquez (passing motorist, line of sight: brief, sober)

I was driving past on {base_loc} around {base_hour}:20 PM. There was a man running across
the street. He was definitely shorter than 6 feet — I'd estimate {alt_height}. He was wearing
a hoodie, gray I think, not dark. I noticed because he almost ran in front of my car. He went
into the alley between the bank and the dry cleaner. I called the police as soon as I parked.""",

        f"""## Witness Statement #6: David Brennan (homeless, line of sight: from across street, slight intoxication noted by responding officer)

I was sitting on the bench across from the place. I seen the whole thing. He went in, then
he came out maybe five minutes later. Couldn't tell you his height — he looked normal, not
short, not tall. He was wearing dark clothes, but I think there was something red about it,
maybe his shoes? Or a logo? It was kinda dark out by then. He had a bag, definitely had a bag
when he left. Walked, didn't run. Walked normally toward {alt_loc}.""",

        f"""## Witness Statement #7: Officer Ramirez (responding officer, arrived 7 minutes after 911 call)

Dispatch received the 911 call at {base_hour}:21 PM. I arrived on scene at {base_hour}:28 PM.
The cashier (Witness #1) was the only person inside the store at the time of my arrival. She
provided an initial description matching: white male, approximately {suspect_height}, dark
jacket, baseball cap, fled with a black backpack. The interior security camera was operational
and timestamped his entry at {base_hour}:14 PM and exit at {base_hour}:17 PM. No other physical
evidence was recovered at the scene at that time.""",

        f"""## Witness Statement #8: Jennifer Wu (next-door coffee shop worker, line of sight: window view of front entrance, sober)

I was wiping down tables near the front window of the coffee shop. I have a clear view of the
neighboring store's entrance from there. I saw a man enter at — I checked my phone right after —
{base_hour}:14 PM. He was wearing a dark blue or navy jacket, definitely not gray, definitely not
a hoodie — it had a collar. He was tall, taller than me by a lot, and I'm 5'6". He came out
about three minutes later carrying a black backpack, which he had NOT been carrying when he went
in. He turned right and walked toward {base_loc}, NOT into any alley.""",
    ]

    return "\n\n".join(statements)


def generate_email_thread_section() -> str:
    """Six-month email thread about a notification system feature with scope creep.

    Supports t3_q3_evolving_requirements: chronological email thread tracking
    the original requirement, multiple changes with dates+requesters, and
    a scope-creep inflection point. Provides material for original-requirement
    identification, change tracking, gap analysis, and final-implementation comparison.
    """
    base_year = 2024
    senders = ["Sarah Mitchell (Product Manager)",
               "Tom Hansen (Engineering Lead)",
               "Priya Desai (UX Designer)",
               "Marcus Webb (CEO)",
               "Sarah Mitchell (Product Manager)",
               "Tom Hansen (Engineering Lead)",
               "Sarah Mitchell (Product Manager)",
               "Marcus Webb (CEO)",
               "Tom Hansen (Engineering Lead)",
               "Sarah Mitchell (Product Manager)",
               "Priya Desai (UX Designer)",
               "Tom Hansen (Engineering Lead)"]

    emails = [
        f"""## Email 1
From: Sarah Mitchell (Product Manager)
To: Engineering, Design
Date: {base_year}-01-15 09:23 AM
Subject: New feature request: Notification system

Hi team,

We need a notification system for the dashboard. Original requirement is straightforward:

- Email notification to users when a workflow completes
- Single configurable preference: on/off
- One template, no customization
- Send within 5 minutes of completion event

Goal: ship in next sprint (4 weeks). This should be a simple feature.

Let me know if you have questions.

Sarah""",

        f"""## Email 2
From: Tom Hansen (Engineering Lead)
To: Sarah Mitchell, Engineering
Date: {base_year}-01-22 02:14 PM
Subject: Re: New feature request: Notification system

Sarah,

Sounds good. We can wire this up using the existing email service in about a week of dev
plus QA. I have one question: should this work for failed workflows too, or just successful
completion? Failed workflows have different metadata available.

Tom""",

        f"""## Email 3
From: Sarah Mitchell (Product Manager)
To: Tom Hansen, Engineering
Date: {base_year}-01-23 10:08 AM
Subject: Re: New feature request: Notification system

Tom — good question. Let's include failed workflows too. So that's two notification types:
"workflow completed" and "workflow failed". Each with its own template. Still on/off as one
toggle.

Sarah""",

        f"""## Email 4
From: Priya Desai (UX Designer)
To: Sarah Mitchell, Tom Hansen
Date: {base_year}-02-08 11:45 AM
Subject: Re: New feature request: Notification system

Hi both,

Coming back from user research — users actually want more granularity than on/off. They want
to choose which workflow types trigger notifications. Could we make it per-workflow-type?
Otherwise we'll get complaints about notification spam.

Priya""",

        f"""## Email 5
From: Marcus Webb (CEO)
To: Sarah Mitchell
Cc: Tom Hansen, Priya Desai
Date: {base_year}-02-19 04:30 PM
Subject: Re: New feature request: Notification system

Sarah —

Just got off the phone with the customer advisory board. Two enterprise customers asked
about Slack integration for these notifications. Can we add Slack as a channel option in
addition to email? They're asking specifically and I'd like to say yes before the renewal
conversation next month.

Marcus""",

        f"""## Email 6
From: Sarah Mitchell (Product Manager)
To: Marcus Webb, Tom Hansen, Priya Desai
Date: {base_year}-02-20 09:11 AM
Subject: Re: New feature request: Notification system

Adding Slack to the scope. So now we have:
- Email + Slack channels
- Per-workflow-type granularity
- Two notification types (success/failure)

Tom — what's the new estimate?

Sarah""",

        f"""## Email 7
From: Tom Hansen (Engineering Lead)
To: Sarah Mitchell, Marcus Webb
Date: {base_year}-02-21 03:55 PM
Subject: Re: New feature request: Notification system

Hi Sarah, Marcus,

The new scope is a different system. We're now talking 6-8 weeks instead of 4. Slack
integration alone is 2 weeks (OAuth flow, channel selection UI, message formatting).
Per-workflow granularity adds another preference UI. Two templates per channel = 4 templates.

Should we still ship Q1?

Tom""",

        f"""## Email 8
From: Marcus Webb (CEO)
To: Tom Hansen, Sarah Mitchell
Date: {base_year}-03-04 06:22 PM
Subject: Re: New feature request: Notification system

Tom, Sarah —

Need this by end of Q1. Customer advisory board is the priority. Do whatever it takes.
We can also add SMS — one of the customers brought it up. Add it to the scope.

Marcus

[NOTE: this email is widely cited internally as the moment scope-creep became
irreversible — the SMS addition without engineering pushback locked in a fundamentally
different system architecture.]""",

        f"""## Email 9
From: Tom Hansen (Engineering Lead)
To: Sarah Mitchell, Marcus Webb
Date: {base_year}-03-08 10:47 AM
Subject: Re: New feature request: Notification system

To make Q1 work, I need to bring in two contractors. Total cost ~$40K. Also need to push
the OAuth library upgrade to Q2 because we don't have time to do both. Approve?

Tom""",

        f"""## Email 10
From: Sarah Mitchell (Product Manager)
To: Tom Hansen
Date: {base_year}-03-15 08:30 AM
Subject: Re: New feature request: Notification system

Tom — also adding mobile push notifications. The mobile team said it's "a quick wire-up."

Sarah""",

        f"""## Email 11
From: Priya Desai (UX Designer)
To: Sarah Mitchell, Tom Hansen
Date: {base_year}-04-02 02:00 PM
Subject: Re: New feature request: Notification system

I've now redesigned the preferences UI three times. Could we freeze the design? Each scope
addition (Slack, SMS, push) means rework. We're at 14 different toggles now.

Priya""",

        f"""## Email 12
From: Tom Hansen (Engineering Lead)
To: Sarah Mitchell, Marcus Webb, Priya Desai
Date: {base_year}-06-28 05:45 PM
Subject: Re: New feature request: Notification system — SHIPPED

Shipped today after 5+ months. Final feature set:
- 4 channels: Email, Slack, SMS, Mobile Push
- Per-workflow-type granularity (12 workflow types × 4 channels = 48 toggles)
- Three notification types (success/failure/started — added in May at customer request)
- 12 templates total
- Custom Slack channel routing
- SMS rate limiting per user

Original Jan estimate: 4 weeks. Actual: 23 weeks. Original requirement was 1 toggle for
1 channel for 1 event type. Final implementation is 48 toggles for 4 channels for 3 event
types. We never circled back to reconcile the in-app notification feature that customers
keep asking for and that everyone assumed would be obvious — there's still no in-app bell icon.

Tom""",
    ]

    return "\n\n".join(emails)


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
    # 2026-05-06: Tier-3 long_context contexts. Pre-2026-05-06 these silently fell
    # back to generate_tech_doc_section, which couldn't support multi-hop reasoning,
    # contradiction detection, or requirement evolution tasks. All models correctly
    # refused on the resulting tech-docs context, making the tier-3 questions
    # unable to discriminate model quality.
    "investigation_docs": generate_investigation_docs_section,
    "witness_statements": generate_witness_statements_section,
    "email_thread": generate_email_thread_section,
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
