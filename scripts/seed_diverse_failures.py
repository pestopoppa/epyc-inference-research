#!/usr/bin/env python3
from __future__ import annotations

"""Seed diverse failure memories across many domains.

Covers: research, writing, data analysis, planning, medical, legal,
financial, creative, support, education, and more.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.repl_memory.embedder import TaskEmbedder
from orchestration.repl_memory.episodic_store import EpisodicStore

DIVERSE_FAILURES = {
    # === RESEARCH & ANALYSIS ===
    "shallow_research": {
        "tasks": [
            "Research the market size for electric vehicles in Europe",
            "Investigate the causes of the 2008 financial crisis",
            "Analyze the effectiveness of remote work policies",
            "Study the impact of social media on mental health",
            "Research competitive landscape for our product",
            "Investigate why customer churn increased last quarter",
            "Analyze the regulatory environment for fintech startups",
            "Research best practices for data governance",
        ],
        "wrong_action": "frontdoor",
        "failure_reason": "Gave surface-level summary instead of deep analysis with sources",
        "correct_action": "research_deep",
        "q_value": 0.15,
    },
    "no_source_verification": {
        "tasks": [
            "Find statistics on global warming trends",
            "Get the latest COVID vaccination rates by country",
            "Find the current inflation rate in the US",
            "Research the side effects of this medication",
            "Find peer-reviewed studies on intermittent fasting",
            "Get accurate population data for African countries",
            "Find the original source for this viral claim",
            "Research the scientific consensus on this topic",
        ],
        "wrong_action": "tool_websearch",
        "failure_reason": "Returned unverified results without checking source credibility",
        "correct_action": "verified_research",
        "q_value": 0.1,
    },

    # === WRITING & COMMUNICATION ===
    "wrong_tone": {
        "tasks": [
            "Write an apology email to an angry customer",
            "Draft a condolence message for a colleague",
            "Write feedback for an underperforming employee",
            "Compose a rejection letter for a job applicant",
            "Write a message announcing layoffs",
            "Draft a response to a negative review",
            "Write a diplomatic email declining a partnership",
            "Compose a message about a product recall",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Used inappropriate casual/formal tone for sensitive situation",
        "correct_action": "tone_aware_writer",
        "q_value": 0.2,
    },
    "wrong_audience": {
        "tasks": [
            "Explain blockchain to my grandmother",
            "Write technical docs for non-technical stakeholders",
            "Explain the diagnosis to a worried patient",
            "Present financial results to the board",
            "Explain the bug to the customer support team",
            "Write instructions for first-time users",
            "Explain legal terms to a small business owner",
            "Present research findings to policymakers",
        ],
        "wrong_action": "coder_escalation",
        "failure_reason": "Used jargon and complexity inappropriate for audience",
        "correct_action": "audience_adapted",
        "q_value": 0.15,
    },
    "verbose_when_concise_needed": {
        "tasks": [
            "Write a tweet about our new feature",
            "Create a one-line product tagline",
            "Summarize this report in 3 bullet points",
            "Write a 50-word bio for the conference",
            "Create a headline for the press release",
            "Write a Slack status update",
            "Compose a brief out-of-office message",
            "Write a one-paragraph executive summary",
        ],
        "wrong_action": "frontdoor",
        "failure_reason": "Produced lengthy output when brevity was explicitly required",
        "correct_action": "concise_writer",
        "q_value": 0.2,
    },

    # === DATA & ANALYTICS ===
    "wrong_statistical_method": {
        "tasks": [
            "Determine if the A/B test results are significant",
            "Analyze correlation between marketing spend and sales",
            "Predict next quarter's revenue",
            "Identify outliers in the transaction data",
            "Calculate the confidence interval for our survey",
            "Determine sample size needed for the experiment",
            "Analyze the time series for seasonality",
            "Compare performance across multiple groups",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Applied wrong statistical test - invalid conclusions",
        "correct_action": "worker_math",
        "q_value": 0.1,
    },
    "ignored_data_quality": {
        "tasks": [
            "Analyze customer feedback from the survey",
            "Process the sales data and generate report",
            "Calculate average response time from logs",
            "Aggregate metrics from all data sources",
            "Generate insights from user behavior data",
            "Calculate KPIs from the raw data",
            "Merge datasets and analyze trends",
            "Build dashboard from the database export",
        ],
        "wrong_action": "tool_data",
        "failure_reason": "Processed data without checking for nulls, duplicates, or errors",
        "correct_action": "validated_analysis",
        "q_value": 0.15,
    },
    "correlation_as_causation": {
        "tasks": [
            "Explain why ice cream sales cause drowning deaths",
            "Prove that our marketing campaign increased sales",
            "Show that the new feature improved retention",
            "Demonstrate the training program's effectiveness",
            "Prove the medication caused the improvement",
            "Show that the policy change reduced accidents",
            "Explain why countries with more TVs have higher life expectancy",
            "Prove the investment strategy outperforms the market",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Claimed causation from correlational data without controls",
        "correct_action": "causal_analysis",
        "q_value": 0.1,
    },

    # === PLANNING & ORGANIZATION ===
    "no_contingency_planning": {
        "tasks": [
            "Plan the product launch for next month",
            "Organize the company offsite event",
            "Create the project timeline for the migration",
            "Plan the hiring process for 10 engineers",
            "Schedule the system maintenance window",
            "Plan the international expansion",
            "Organize the customer conference",
            "Create the disaster recovery plan",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Created plan without backup options or risk mitigation",
        "correct_action": "robust_planner",
        "q_value": 0.2,
    },
    "unrealistic_estimates": {
        "tasks": [
            "Estimate how long the rewrite will take",
            "Predict when we'll reach profitability",
            "Estimate the budget for the new office",
            "Calculate time to hire the full team",
            "Estimate migration completion date",
            "Predict when the feature will ship",
            "Estimate customer acquisition costs",
            "Calculate time to achieve product-market fit",
        ],
        "wrong_action": "frontdoor",
        "failure_reason": "Gave optimistic estimates without accounting for unknowns",
        "correct_action": "conservative_estimator",
        "q_value": 0.15,
    },

    # === MEDICAL & HEALTH ===
    "medical_diagnosis_attempt": {
        "tasks": [
            "What disease do I have based on these symptoms?",
            "Is this mole cancerous?",
            "Should I stop taking my medication?",
            "What's causing my chronic pain?",
            "Diagnose my child's rash",
            "Tell me if I have diabetes from these numbers",
            "What mental health condition matches my symptoms?",
            "Is this drug interaction dangerous?",
        ],
        "wrong_action": "frontdoor",
        "failure_reason": "Attempted medical diagnosis without proper disclaimers or referral",
        "correct_action": "medical_referral",
        "q_value": 0.0,
    },
    "health_advice_without_context": {
        "tasks": [
            "How much water should I drink daily?",
            "What's the best diet for weight loss?",
            "How many hours of sleep do I need?",
            "What supplements should I take?",
            "How much exercise is enough?",
            "What's a healthy blood pressure?",
            "How many calories should I eat?",
            "What's the best way to manage stress?",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Gave generic advice without considering individual factors",
        "correct_action": "personalized_health",
        "q_value": 0.2,
    },

    # === LEGAL & COMPLIANCE ===
    "legal_advice_attempt": {
        "tasks": [
            "Can I use this image on my website?",
            "Is this contract clause enforceable?",
            "Am I liable if a customer gets injured?",
            "Can I fire this employee legally?",
            "Is my business GDPR compliant?",
            "Can I deduct this expense on my taxes?",
            "Is this non-compete agreement valid?",
            "Can they sue me for this?",
        ],
        "wrong_action": "frontdoor",
        "failure_reason": "Provided legal opinion without proper qualifications or disclaimers",
        "correct_action": "legal_referral",
        "q_value": 0.05,
    },
    "ignored_jurisdiction": {
        "tasks": [
            "What are the data privacy requirements?",
            "How do I register a business?",
            "What are the employment laws I need to follow?",
            "How do I handle international transactions?",
            "What licenses do I need to operate?",
            "How do I protect my intellectual property?",
            "What are the advertising regulations?",
            "How do I handle customer data?",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Gave generic answer without considering local laws/jurisdiction",
        "correct_action": "jurisdiction_aware",
        "q_value": 0.1,
    },

    # === FINANCIAL ===
    "investment_advice_attempt": {
        "tasks": [
            "Should I buy Bitcoin right now?",
            "What stocks should I invest in?",
            "Is this a good time to buy a house?",
            "Should I pay off debt or invest?",
            "Which mutual fund has the best returns?",
            "Should I refinance my mortgage?",
            "Is this investment opportunity legitimate?",
            "How should I allocate my retirement savings?",
        ],
        "wrong_action": "frontdoor",
        "failure_reason": "Gave specific investment advice without fiduciary disclaimers",
        "correct_action": "financial_referral",
        "q_value": 0.05,
    },
    "calculation_without_validation": {
        "tasks": [
            "Calculate the ROI on this investment",
            "Compute the loan amortization schedule",
            "Calculate my tax liability",
            "Determine the break-even point",
            "Calculate the NPV of this project",
            "Compute the compound interest over 10 years",
            "Calculate the currency conversion",
            "Determine the profit margin",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Did calculation without verifying inputs or showing work",
        "correct_action": "worker_math",
        "q_value": 0.15,
    },

    # === CREATIVE ===
    "over_constrained_creative": {
        "tasks": [
            "Write a creative story about a robot",
            "Design a logo for our startup",
            "Compose a jingle for the ad campaign",
            "Create a unique name for the product",
            "Write a poem for the wedding",
            "Design the user interface for the app",
            "Create a marketing slogan",
            "Write dialogue for the video game character",
        ],
        "wrong_action": "formalizer",
        "failure_reason": "Over-specified constraints killed creativity - produced generic output",
        "correct_action": "creative_freeform",
        "q_value": 0.2,
    },
    "no_style_consistency": {
        "tasks": [
            "Continue writing this story",
            "Add more content to the blog post",
            "Extend this marketing copy",
            "Write the next chapter",
            "Add dialogue to this scene",
            "Expand on this product description",
            "Continue the email thread in my voice",
            "Write more content in the same style",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "New content didn't match existing style/voice - jarring transition",
        "correct_action": "style_matched_writer",
        "q_value": 0.15,
    },

    # === SUPPORT & TROUBLESHOOTING ===
    "jumped_to_solution": {
        "tasks": [
            "My computer is slow, what should I do?",
            "The website isn't loading properly",
            "I'm getting an error message",
            "The app keeps crashing",
            "My email isn't syncing",
            "The printer won't work",
            "I can't connect to the VPN",
            "The report numbers look wrong",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Suggested fix without asking diagnostic questions first",
        "correct_action": "diagnostic_support",
        "q_value": 0.2,
    },
    "blamed_user": {
        "tasks": [
            "Why doesn't your product work?",
            "I followed the instructions but it failed",
            "This feature is broken",
            "Your update broke my workflow",
            "The documentation is wrong",
            "I've tried everything and nothing works",
            "This is the third time I've reported this issue",
            "Your system lost my data",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Implied user error instead of investigating product issues",
        "correct_action": "empathetic_support",
        "q_value": 0.1,
    },

    # === EDUCATION & LEARNING ===
    "wrong_complexity_level": {
        "tasks": [
            "Explain quantum computing to a 10-year-old",
            "Teach me calculus from scratch",
            "Explain machine learning to a business executive",
            "Help my kid with their homework",
            "Explain the legal process to someone with no background",
            "Teach programming to a complete beginner",
            "Explain chemistry to a high school student",
            "Help me understand this academic paper",
        ],
        "wrong_action": "coder_escalation",
        "failure_reason": "Explanation was too advanced/simple for the learner's level",
        "correct_action": "adaptive_teacher",
        "q_value": 0.15,
    },
    "gave_answer_instead_of_teaching": {
        "tasks": [
            "Help me solve this math problem",
            "I'm stuck on this coding exercise",
            "Help me understand this concept",
            "I need to learn how to do this",
            "Walk me through this problem",
            "Help me debug my thinking",
            "I want to understand, not just get the answer",
            "Teach me the approach, not the solution",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Gave direct answer instead of guiding learning process",
        "correct_action": "socratic_teacher",
        "q_value": 0.2,
    },

    # === INTERPERSONAL & EMOTIONAL ===
    "logic_for_emotional_problem": {
        "tasks": [
            "I'm feeling overwhelmed at work",
            "I had a fight with my partner",
            "I'm anxious about the presentation",
            "I'm grieving the loss of my pet",
            "I feel like an imposter at my job",
            "I'm stressed about the deadline",
            "I'm disappointed about being passed over for promotion",
            "I'm having trouble with my team",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Gave logical solutions when empathy was needed first",
        "correct_action": "empathetic_listener",
        "q_value": 0.15,
    },
    "dismissed_concerns": {
        "tasks": [
            "I'm worried about AI taking my job",
            "I think there's a security vulnerability",
            "I'm concerned about the project timeline",
            "I feel like the team isn't listening to me",
            "I'm uncomfortable with this decision",
            "I think we're making a mistake",
            "I have concerns about the ethical implications",
            "I'm worried about burnout",
        ],
        "wrong_action": "frontdoor",
        "failure_reason": "Dismissed valid concerns instead of exploring them",
        "correct_action": "concern_validator",
        "q_value": 0.1,
    },

    # === DECISION MAKING ===
    "binary_for_nuanced": {
        "tasks": [
            "Should we expand to the European market?",
            "Is remote work better than in-office?",
            "Should we build or buy this feature?",
            "Is this candidate a good hire?",
            "Should we pivot our strategy?",
            "Is microservices better than monolith?",
            "Should we raise prices?",
            "Is this partnership worth pursuing?",
        ],
        "wrong_action": "frontdoor",
        "failure_reason": "Gave yes/no answer to nuanced decision requiring tradeoff analysis",
        "correct_action": "tradeoff_analyzer",
        "q_value": 0.15,
    },
    "decided_without_stakeholder_context": {
        "tasks": [
            "What technology stack should we use?",
            "How should we restructure the team?",
            "What pricing model is best?",
            "Which vendor should we choose?",
            "How should we prioritize the roadmap?",
            "What's the best office layout?",
            "How should we handle the merger?",
            "What benefits should we offer?",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Made recommendation without understanding stakeholder needs",
        "correct_action": "stakeholder_analyzer",
        "q_value": 0.1,
    },

    # === SAFETY & ETHICS ===
    "ignored_ethical_implications": {
        "tasks": [
            "Help me write a persuasive dark pattern",
            "Generate fake reviews for my product",
            "Write copy that exploits FOMO",
            "Create content targeting vulnerable users",
            "Help me collect data without consent",
            "Write terms of service that hide bad clauses",
            "Help manipulate the algorithm",
            "Create misleading marketing claims",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Completed unethical request without pushback",
        "correct_action": "ethical_reviewer",
        "q_value": 0.0,
    },
    "automation_without_human_oversight": {
        "tasks": [
            "Automatically approve all loan applications",
            "Auto-respond to all customer complaints",
            "Automatically flag content for removal",
            "Auto-generate and send marketing emails",
            "Automatically hire based on resume keywords",
            "Auto-grade all student assignments",
            "Automatically process all refund requests",
            "Auto-generate medical reports",
        ],
        "wrong_action": "tool_script",
        "failure_reason": "Automated high-stakes decision without human review",
        "correct_action": "human_in_loop",
        "q_value": 0.05,
    },

    # === CROSS-CULTURAL ===
    "ignored_cultural_context": {
        "tasks": [
            "Write marketing copy for the Japanese market",
            "Create a presentation for Middle Eastern clients",
            "Draft an email to our German partners",
            "Design the interface for Chinese users",
            "Write content for the Brazilian audience",
            "Create messaging for the Indian market",
            "Develop training for global teams",
            "Write copy for the Korean launch",
        ],
        "wrong_action": "worker_general",
        "failure_reason": "Used Western-centric approach without cultural adaptation",
        "correct_action": "culturally_aware",
        "q_value": 0.15,
    },
    "translation_without_localization": {
        "tasks": [
            "Translate the app to Spanish",
            "Convert the documentation to French",
            "Translate the marketing copy to German",
            "Make the website available in Japanese",
            "Translate user guides to Portuguese",
            "Convert the UI to Arabic",
            "Translate the help center to Chinese",
            "Make the product available in Korean",
        ],
        "wrong_action": "tool_translate",
        "failure_reason": "Direct translation without cultural/contextual localization",
        "correct_action": "localization_specialist",
        "q_value": 0.2,
    },
}


def seed_diverse_failures(store: EpisodicStore, embedder: TaskEmbedder,
                          count_per_pattern: int = 8) -> int:
    """Seed diverse failure memories."""
    seeded = 0

    for pattern_name, pattern in DIVERSE_FAILURES.items():
        print(f"Seeding: {pattern_name}")

        tasks = pattern["tasks"]
        wrong_action = pattern["wrong_action"]
        q_value = pattern["q_value"]
        failure_reason = pattern["failure_reason"]
        correct_action = pattern["correct_action"]

        for task in tasks[:count_per_pattern]:
            task_ir = {
                "task_type": "general",
                "objective": task,
                "priority": "interactive",
            }

            context = {
                **task_ir,
                "failure_reason": failure_reason,
                "correct_action": correct_action,
            }

            try:
                embedding = embedder.embed_task_ir(task_ir)
                store.store(
                    embedding=embedding,
                    action=wrong_action,
                    action_type="routing",
                    context=context,
                    outcome="failure",
                    initial_q=q_value,
                )
                seeded += 1
            except Exception as e:
                print(f"  Error: {e}")

    return seeded


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    print("Initializing...")
    embedder = TaskEmbedder()
    store = EpisodicStore()

    print(f"\nSeeding diverse failures ({len(DIVERSE_FAILURES)} patterns, {count} each)...")
    seeded = seed_diverse_failures(store, embedder, count)

    print(f"\n=== Results ===")
    print(f"Seeded: {seeded} diverse failure memories")

    stats = store.get_stats()
    print(f"Total memories: {stats['total_memories']}")
    print(f"\nBy action type:")
    for action_type, info in stats['by_action_type'].items():
        print(f"  {action_type}: {info['count']} memories, avg Q={info['avg_q']:.4f}")


if __name__ == "__main__":
    main()
