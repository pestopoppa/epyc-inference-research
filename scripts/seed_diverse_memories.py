#!/usr/bin/env python3
from __future__ import annotations

"""Seed diverse memories for different task domains.

Creates synthetic memory entries for math, medical, workflow, coding, etc.
to help the system learn appropriate routing for different task types.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.embedder import TaskEmbedder
from orchestration.repl_memory.episodic_store import EpisodicStore

# Domain-specific task templates
TASK_TEMPLATES = {
    "architect_design": {
        "task_type": "design",
        "routing": "architect_general",
        "templates": [
            "Design a scalable architecture for {system}",
            "What are the trade-offs between {option1} and {option2} for {context}",
            "Review the system design for {system} and identify potential issues",
            "Create a high-level architecture diagram for {system}",
            "Evaluate whether we should use {tech} for {use_case}",
            "Design the data model for {domain}",
            "What's the best approach to handle {scale} concurrent users",
            "Design a disaster recovery strategy for {system}",
            "Architect a migration path from {old_tech} to {new_tech}",
            "Design the security architecture for {system}",
            "Evaluate microservices vs monolith for {project}",
            "Design the API versioning strategy for {api}",
        ],
        "vars": {
            "system": ["e-commerce platform", "real-time chat system", "data pipeline",
                      "authentication service", "recommendation engine", "payment gateway"],
            "option1": ["SQL", "Redis", "Kafka", "REST", "GraphQL", "microservices"],
            "option2": ["NoSQL", "Memcached", "RabbitMQ", "gRPC", "REST", "monolith"],
            "context": ["high write throughput", "low latency reads", "event sourcing",
                       "mobile apps", "internal APIs", "startup with 10 engineers"],
            "tech": ["Kubernetes", "serverless", "event sourcing", "CQRS", "GraphQL"],
            "use_case": ["scaling to 1M users", "real-time notifications", "ML inference"],
            "domain": ["multi-tenant SaaS", "healthcare records", "financial transactions"],
            "scale": ["10K", "100K", "1M", "10M"],
            "old_tech": ["PHP monolith", "on-prem", "MySQL"],
            "new_tech": ["Go microservices", "cloud-native", "PostgreSQL"],
            "project": ["early-stage startup", "enterprise app", "high-traffic API"],
            "api": ["public REST API", "partner integrations", "mobile backend"],
        },
    },
    "architect_code": {
        "task_type": "code",
        "routing": "architect_coding",
        "templates": [
            "Design the class hierarchy for {domain}",
            "What design pattern should we use for {problem}",
            "Review this architecture: {description}",
            "How should we structure the {component} module",
            "Design the interface between {component1} and {component2}",
            "Create an abstraction for {concept}",
            "Design the error handling strategy for {system}",
            "What's the best way to implement {feature} given {constraints}",
        ],
        "vars": {
            "domain": ["plugin system", "state management", "data access layer",
                      "event handling", "caching layer", "authentication flow"],
            "problem": ["handling multiple payment providers", "supporting multiple databases",
                       "implementing undo/redo", "managing complex state transitions"],
            "description": ["repository pattern with CQRS", "hexagonal architecture",
                           "event-driven microservices", "modular monolith"],
            "component": ["core", "plugins", "adapters", "services", "utilities"],
            "component1": ["API gateway", "business logic", "data layer"],
            "component2": ["external services", "caching", "message queue"],
            "concept": ["database connections", "HTTP clients", "file storage"],
            "system": ["distributed system", "real-time app", "batch processor"],
            "feature": ["rate limiting", "circuit breaker", "retry logic", "caching"],
            "constraints": ["no external dependencies", "must be synchronous", "memory constraints"],
        },
    },
    "formalize_vague": {
        "task_type": "formalize",
        "routing": "formalizer",
        "templates": [
            "Help me with {vague_task}",
            "I need to {vague_goal}",
            "Can you work on {vague_request}",
            "Figure out how to {vague_objective}",
            "Make {vague_thing} better",
            "Something's wrong with {vague_component}",
            "I want to improve {vague_area}",
            "Do something about {vague_issue}",
            "Handle {vague_situation}",
            "Take care of {vague_task}",
        ],
        "vars": {
            "vague_task": ["the deployment thing", "that bug we discussed", "the performance stuff",
                         "the API issues", "user complaints", "the code quality"],
            "vague_goal": ["fix things", "make it faster", "improve the system",
                         "clean up the code", "update the docs", "modernize our stack"],
            "vague_request": ["the project", "the app", "the backend", "our infrastructure"],
            "vague_objective": ["optimize performance", "improve UX", "reduce costs",
                              "scale better", "be more reliable"],
            "vague_thing": ["the code", "the system", "the tests", "the deployment"],
            "vague_component": ["the API", "the database", "the frontend", "the pipeline"],
            "vague_area": ["security", "performance", "reliability", "developer experience"],
            "vague_issue": ["the slowness", "the errors", "the complaints", "the tech debt"],
            "vague_situation": ["the outage prep", "the migration", "the upgrade"],
        },
    },
    "tool_websearch": {
        "task_type": "tool",
        "routing": "tool_websearch",
        "templates": [
            "Search Wikipedia for {topic}",
            "What's the current weather in {location}",
            "Find recent news about {subject}",
            "Look up the definition of {term}",
            "Get the latest {data_type} for {entity}",
            "Search for {query} on {source}",
            "Find information about {topic} from {timeframe}",
            "What are the top results for {search_query}",
        ],
        "vars": {
            "topic": ["quantum computing", "French Revolution", "machine learning",
                     "climate change", "cryptocurrency", "protein folding"],
            "location": ["San Francisco", "Tokyo", "London", "Sydney", "Mumbai"],
            "subject": ["AI developments", "tech earnings", "space exploration",
                       "electric vehicles", "renewable energy"],
            "term": ["epistemology", "homeostasis", "entropy", "recursion"],
            "data_type": ["stock price", "population", "GDP", "temperature"],
            "entity": ["Apple", "Japan", "Bitcoin", "Amazon"],
            "source": ["arxiv", "HackerNews", "Reddit", "GitHub"],
            "timeframe": ["today", "this week", "2024", "the last hour"],
            "search_query": ["best practices for X", "how to implement Y", "tutorial for Z"],
        },
    },
    "tool_github": {
        "task_type": "tool",
        "routing": "tool_github",
        "templates": [
            "Clone and analyze the {repo} repository",
            "Find all open issues in {repo}",
            "Get the recent commits from {repo}",
            "Search for {pattern} in {repo}",
            "List the contributors to {repo}",
            "Find repos similar to {repo}",
            "Get the release notes for {repo}",
            "Check the CI status for {repo}",
        ],
        "vars": {
            "repo": ["facebook/react", "microsoft/vscode", "torvalds/linux",
                    "python/cpython", "rust-lang/rust", "kubernetes/kubernetes"],
            "pattern": ["security vulnerabilities", "deprecated functions",
                       "TODO comments", "error handling patterns"],
        },
    },
    "tool_script": {
        "task_type": "tool",
        "routing": "tool_script",
        "templates": [
            "Run a Monte Carlo simulation for {problem}",
            "Calculate {computation} using numerical methods",
            "Generate {count} random {items} matching {criteria}",
            "Benchmark {operation} with {parameters}",
            "Profile the performance of {code}",
            "Generate test data for {schema}",
            "Validate {data} against {rules}",
            "Convert {input_format} to {output_format}",
        ],
        "vars": {
            "problem": ["pi estimation", "option pricing", "risk assessment", "queue wait times"],
            "computation": ["the 1000th prime", "factorial of 100", "Fibonacci sequence"],
            "count": ["100", "1000", "10000"],
            "items": ["user records", "transaction logs", "test cases", "API requests"],
            "criteria": ["valid email addresses", "US phone numbers", "dates in 2024"],
            "operation": ["sorting algorithm", "database query", "API endpoint", "regex matching"],
            "parameters": ["different array sizes", "varying concurrency", "multiple iterations"],
            "code": ["the main loop", "database queries", "image processing"],
            "schema": ["user table", "order system", "inventory management"],
            "data": ["JSON payload", "CSV file", "API response"],
            "rules": ["schema validation", "business rules", "format requirements"],
            "input_format": ["CSV", "XML", "YAML"],
            "output_format": ["JSON", "Parquet", "SQL"],
        },
    },
    "math": {
        "task_type": "math",
        "routing": "worker_math",
        "templates": [
            "Solve the integral of {func} dx",
            "Prove that {theorem}",
            "Calculate the derivative of {func}",
            "Find the eigenvalues of matrix {matrix}",
            "Solve the differential equation {equation}",
            "Evaluate the limit as x approaches {value} of {func}",
            "Factor the polynomial {poly}",
            "Simplify the expression {expr}",
            "Calculate the probability of {event}",
            "Determine if {series} converges or diverges",
            "Find the Taylor series expansion of {func}",
            "Compute the Laplace transform of {func}",
            "Solve the system of linear equations {system}",
            "Calculate the expected value of {distribution}",
            "Prove by induction that {statement}",
        ],
        "vars": {
            "func": ["sin(x)", "e^x", "ln(x)", "x^3 - 2x + 1", "1/(1+x^2)", "sqrt(x)", "tan(x)"],
            "theorem": ["sqrt(2) is irrational", "there are infinitely many primes",
                       "e is transcendental", "the sum of angles in a triangle is 180 degrees"],
            "matrix": ["[[1,2],[3,4]]", "[[a,b],[c,d]]", "3x3 rotation matrix"],
            "equation": ["dy/dx = y", "y'' + y = 0", "x*dy/dx - y = x^2"],
            "value": ["0", "infinity", "1", "pi"],
            "poly": ["x^4 - 1", "x^3 - 6x^2 + 11x - 6", "x^2 + 5x + 6"],
            "expr": ["(a+b)^3", "sin^2(x) + cos^2(x)", "log_a(b) * log_b(a)"],
            "event": ["rolling a 7 with two dice", "getting 3 heads in 5 coin flips"],
            "series": ["sum of 1/n^2", "sum of 1/n", "sum of (-1)^n/n"],
            "distribution": ["binomial(n,p)", "Poisson(lambda)", "normal(mu, sigma)"],
            "statement": ["n! > 2^n for n >= 4", "sum of first n odd numbers = n^2"],
            "system": ["2x + 3y = 7, x - y = 1", "x + y + z = 6, 2x - y = 3, y + z = 4"],
        },
    },
    "medical": {
        "task_type": "doc",
        "routing": "frontdoor",
        "templates": [
            "What are the symptoms of {condition}",
            "What is the treatment protocol for {condition}",
            "Explain the mechanism of action of {drug}",
            "What are the contraindications for {drug}",
            "Differential diagnosis for patient presenting with {symptoms}",
            "What lab tests are indicated for {condition}",
            "Explain the pathophysiology of {condition}",
            "What are the risk factors for developing {condition}",
            "Drug interactions between {drug} and {drug2}",
            "First-line treatment for {condition}",
            "Prognosis for patients with {condition}",
            "Staging criteria for {condition}",
            "What imaging studies are recommended for {symptoms}",
            "Interpretation of {lab_result} in context of {symptoms}",
        ],
        "vars": {
            "condition": ["type 2 diabetes", "hypertension", "COPD", "heart failure",
                         "rheumatoid arthritis", "major depressive disorder", "GERD",
                         "chronic kidney disease", "hypothyroidism", "pneumonia"],
            "drug": ["metformin", "lisinopril", "atorvastatin", "omeprazole", "sertraline",
                    "levothyroxine", "amlodipine", "gabapentin", "metoprolol"],
            "drug2": ["aspirin", "warfarin", "ibuprofen", "acetaminophen", "prednisone"],
            "symptoms": ["chest pain and shortness of breath", "fatigue and weight gain",
                        "joint pain and morning stiffness", "persistent cough",
                        "abdominal pain and bloating", "headache and visual changes"],
            "lab_result": ["elevated TSH", "HbA1c of 9%", "BUN/creatinine ratio 25:1",
                          "positive ANA", "elevated troponin"],
        },
    },
    "workflow": {
        "task_type": "manage",
        "routing": "frontdoor",
        "templates": [
            "Create a project plan for {project}",
            "Prioritize these tasks: {tasks}",
            "Schedule meetings for {team} over the next {period}",
            "Break down {feature} into actionable subtasks",
            "Create a timeline for {deliverable}",
            "Allocate resources for {project}",
            "Identify dependencies between {tasks}",
            "Create a risk assessment for {project}",
            "Define milestones for {project}",
            "Estimate effort for {feature}",
            "Create a sprint backlog from {requirements}",
            "Organize retrospective topics for {sprint}",
            "Plan capacity for {team} next quarter",
            "Create acceptance criteria for {story}",
        ],
        "vars": {
            "project": ["mobile app launch", "database migration", "API redesign",
                       "security audit", "performance optimization", "new feature rollout"],
            "tasks": ["bug fixes, new features, testing, documentation",
                     "design review, implementation, code review, deployment"],
            "team": ["engineering team", "product team", "cross-functional team"],
            "period": ["next week", "this month", "Q2"],
            "feature": ["user authentication", "payment processing", "search functionality",
                       "notification system", "reporting dashboard"],
            "deliverable": ["MVP", "v2.0 release", "security patch", "beta launch"],
            "requirements": ["PRD document", "user stories", "feature requests"],
            "sprint": ["Sprint 12", "current sprint", "last release"],
            "story": ["user login", "checkout flow", "admin dashboard"],
        },
    },
    "coding": {
        "task_type": "code",
        "routing": "coder_escalation",
        "templates": [
            "Implement {algorithm} in {language}",
            "Refactor {component} to use {pattern}",
            "Write unit tests for {function}",
            "Debug the {error} in {component}",
            "Optimize {function} for {metric}",
            "Add {feature} to {component}",
            "Convert {code} from {lang1} to {lang2}",
            "Review and improve {code_type}",
            "Fix security vulnerability in {component}",
            "Add error handling to {function}",
            "Create API endpoint for {operation}",
            "Implement caching for {component}",
            "Write migration script for {change}",
            "Add logging and monitoring to {service}",
        ],
        "vars": {
            "algorithm": ["quicksort", "binary search", "BFS/DFS", "dynamic programming solution",
                         "hash table", "priority queue", "LRU cache", "bloom filter"],
            "language": ["Python", "TypeScript", "Rust", "Go", "Java"],
            "component": ["user service", "authentication module", "data pipeline",
                         "API gateway", "worker process", "cache layer"],
            "pattern": ["dependency injection", "factory pattern", "observer pattern",
                       "strategy pattern", "repository pattern"],
            "function": ["parse_config()", "validate_input()", "process_batch()",
                        "serialize_response()", "handle_error()"],
            "error": ["race condition", "memory leak", "null pointer", "off-by-one error"],
            "metric": ["memory usage", "latency", "throughput", "CPU usage"],
            "feature": ["pagination", "filtering", "sorting", "validation", "caching"],
            "code": ["API handlers", "database queries", "test suite"],
            "code_type": ["error handling", "type annotations", "documentation"],
            "lang1": ["JavaScript", "Python 2"],
            "lang2": ["TypeScript", "Python 3"],
            "operation": ["CRUD operations", "search", "bulk import", "export"],
            "change": ["schema update", "data format change", "table rename"],
            "service": ["payment service", "notification service", "auth service"],
        },
    },
    "ingest": {
        "task_type": "ingest",
        "routing": "ingest_long_context",
        "templates": [
            "Summarize the following document: {doc_type}",
            "Extract key points from {content}",
            "Analyze the {doc_type} and provide insights",
            "Compare and contrast {items}",
            "Create an executive summary of {report}",
            "Extract entities and relationships from {content}",
            "Identify themes in {content}",
            "Generate Q&A pairs from {doc_type}",
        ],
        "vars": {
            "doc_type": ["research paper", "legal document", "financial report",
                        "technical specification", "meeting transcript", "email thread"],
            "content": ["the attached text", "these meeting notes", "this code review",
                       "the provided dataset", "these customer reviews"],
            "items": ["two proposals", "competing approaches", "quarterly results"],
            "report": ["annual report", "project status update", "market analysis"],
        },
    },
}


def generate_task(domain: str) -> tuple[dict, str]:
    """Generate a synthetic task for a domain."""
    config = TASK_TEMPLATES[domain]
    template = random.choice(config["templates"])

    # Fill in template variables
    objective = template
    for var, options in config["vars"].items():
        placeholder = "{" + var + "}"
        if placeholder in objective:
            objective = objective.replace(placeholder, random.choice(options), 1)

    task_ir = {
        "task_type": config["task_type"],
        "objective": objective,
        "priority": random.choice(["interactive", "batch"]),
    }

    return task_ir, config["routing"]


def seed_diverse_memories(store: EpisodicStore, embedder: TaskEmbedder,
                         count_per_domain: int = 50) -> int:
    """Seed memories for each domain."""
    seeded = 0

    for domain in TASK_TEMPLATES:
        print(f"\nSeeding {count_per_domain} memories for domain: {domain}")

        for i in range(count_per_domain):
            task_ir, routing = generate_task(domain)

            # Most tasks succeed, but add some failures
            if random.random() < 0.1:
                outcome = "failure"
                initial_q = 0.25
            else:
                outcome = "success"
                initial_q = 1.0

            try:
                embedding = embedder.embed_task_ir(task_ir)
                store.store(
                    embedding=embedding,
                    action=routing,
                    action_type="routing",
                    context=task_ir,
                    outcome=outcome,
                    initial_q=initial_q,
                )
                seeded += 1
            except Exception as e:
                print(f"Error: {e}")
                continue

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{count_per_domain}")

    return seeded


def main():
    count_per_domain = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("Initializing...")
    embedder = TaskEmbedder()
    store = EpisodicStore()

    print(f"Seeding {count_per_domain} memories per domain...")
    seeded = seed_diverse_memories(store, embedder, count_per_domain)

    print(f"\n=== Results ===")
    print(f"Seeded: {seeded} memories")
    stats = store.get_stats()
    print(f"Total memories: {stats['total_memories']}")
    print(f"Avg Q-value: {stats['overall_avg_q']:.4f}")
    print(f"\nBy action type:")
    for action_type, info in stats['by_action_type'].items():
        print(f"  {action_type}: {info['count']} memories, avg Q={info['avg_q']:.4f}")


if __name__ == "__main__":
    main()
