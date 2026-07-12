## Advisor workflow

The main agent is responsible for implementation.

Use the `advisor` subagent before implementation when a task involves:

- architecture or API design;
- changes across several packages or services;
- concurrency, authentication, security, or data consistency;
- a difficult bug whose cause is uncertain;
- migrations or backwards-compatibility risks;
- multiple plausible implementation approaches;
- substantial refactoring.

The advisor must remain read-only.

Workflow:

1. Explore enough to formulate the decision.
2. Ask the advisor for a recommendation.
3. Wait for and evaluate the advisor's response.
4. Implement the selected approach.
5. Run tests and validation.
6. Ask the advisor for a final review when the change is high-risk.
7. Address concrete findings before completing the task.

Do not consult the advisor for trivial, mechanical, or easily reversible work.
