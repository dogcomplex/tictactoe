# REQUIREMENTS.md: Pragmatic Semantic Reasoning System

## 1. Introduction

**1.1. Goal:**
To design and implement a modular, extensible, and pragmatic **Semantic Reasoning System**. This system aims to automatically identify underlying patterns, rules, or algorithms present in arbitrary input data, primarily focusing on input-output examples. The system should be capable of generating and validating hypotheses about the data's generating process.

**1.2. Approach:**
The system will be based on a **meta-synthesis / classifier ensemble architecture**. It will orchestrate multiple distinct hypothesis generation and validation strategies (Symbolic Enumeration, SAT Solvers, LLMs, Classifiers, etc.) running in parallel or stages. The design will prioritize pragmatism, flexibility, and leveraging existing successful components from the current codebase.

**1.3. Foundation:**
This project builds upon:
*   The design principles and architectural decisions discussed in `LOCUS2_chat_gemini.md`.
*   Valuable components from the existing codebase, notably the GPU-accelerated hypothesis enumerator (`recipes.py`) and the SAT-based hypothesis system (`few_shot_algs/sat_hypotheses.py`).
*   The TicTacToe domain (`tictactoe.py`) as an initial, rich testbed, including variations like no-diagonal wins to challenge priors.

## 2. Guiding Principles

The system design and implementation will adhere to the following principles derived from `LOCUS2_chat_gemini.md`:

*   **Modularity:** Components (strategies, verifiers, feature extractors) should be distinct and pluggable.
*   **Standardized Interfaces:** Define clear contracts for I/O specifications, program/hypothesis representations, strategy APIs, verification results, and feature formats.
*   **Flexibility & Pragmatism:**
    *   Employ lossy, best-effort conversion between different program/hypothesis representations (e.g., string -> AST -> DSL) where beneficial.
    *   Use layered/tiered approaches for complex tasks like hypothesis equivalence checking, feature extraction, and verification (start simple/fast, escalate if needed).
    *   Implement hybrid push/pull strategy dispatch (target promising strategies first, fall back to broader search).
    *   Adopt fail-soft error handling with clear logging and confidence scoring.
*   **Transparency:** Implement provenance tracking for generated hypotheses and maintain clear, multi-format reporting (structured JSON, human-readable logs).
*   **Extensibility:** The architecture should easily accommodate new reasoning strategies, feature extractors, and data types in the future.
*   **Human-in-the-Loop:** Initially assume human oversight for complex ranking and final validation, designing outputs accordingly.
*   **Performance Awareness:** Incorporate caching mechanisms and leverage efficient implementations (like the existing GPU code).

## 3. System Architecture (Full Scope)

The target system will eventually consist of the following interconnected components:

*   **3.1. Input Orchestrator:**
    *   Accepts reasoning tasks defined in a standard format (e.g., JSON containing inputs, outputs, constraints, metadata).
    *   Parses, validates, and normalizes task specifications.
*   **3.2. Feature Extractor Bank:**
    *   A registry of diverse feature extractors (simple statistical checks, type detectors, structural analyzers, potentially shallow classifiers or LLM-based summarizers).
    *   Runs selected extractors on input data, producing a standardized feature set.
    *   Supports tiered execution (fast features first).
*   **3.3. Strategy Dispatcher:**
    *   Selects and invokes appropriate hypothesis generation strategies based on input features, task metadata, and potentially learned heuristics (Pull-based).
    *   Supports fallback to broader strategy execution (Push-based) if initial attempts fail.
    *   Manages parallel execution and timeouts.
*   **3.4. Strategy Registry:**
    *   A pluggable registry holding instances of various reasoning strategies.
*   **3.5. Hypothesis Generators / Reasoning Strategies:**
    *   Implementations of diverse reasoning approaches conforming to a standard API. Examples:
        *   Symbolic Enumeration (like `recipes.py`)
        *   SAT/SMT Solvers (like `sat_hypotheses.py`)
        *   Inductive Logic Programming (ILP)
        *   LLM-based Few-Shot Induction / Program Generation
        *   Grammar Induction / Regex Synthesis
        *   Specialized Classifiers / Pattern Matchers
*   **3.6. Program Representation Handler:**
    *   Defines a standard `CandidateProgram` or `Hypothesis` representation (containing code/rule, source, confidence, features, provenance, etc.).
    *   Includes best-effort converters between different internal formats (string, AST, DSL, CNF, tensor) used by various strategies.
*   **3.7. Equivalence Checker:**
    *   A layered system to identify potentially duplicate hypotheses.
    *   Uses fast heuristics (hashing, AST normalization) first, potentially escalating to semantic comparison (e.g., expanded I/O testing, LLM similarity). Designed to be conservative (avoid false positives).
*   **3.8. Verification Engine:**
    *   A tiered engine to check candidate hypotheses against provided examples and constraints.
    *   Supports basic I/O matching, property-based testing, potentially symbolic execution or interfacing with formal methods tools.
*   **3.9. Result Aggregator & Ranker:**
    *   Collects verified candidate hypotheses from all strategies.
    *   Deduplicates candidates using the Equivalence Checker.
    *   Scores and ranks candidates based on verification results, complexity, confidence, provenance, and potentially learned ranking models or human feedback.
*   **3.10. Caching Subsystem:**
    *   Caches results of expensive operations (feature extraction, strategy execution, verification) based on task/input fingerprints.
*   **3.11. Provenance Tracker:**
    *   Attaches metadata to each candidate hypothesis tracking its origin (strategy, version), transformations, features used, and verification history.
*   **3.12. Reporting Module:**
    *   Generates outputs in multiple formats:
        *   Structured JSON for machine consumption.
        *   Human-readable Markdown/log summaries for debugging and analysis.

## 4. Analysis of Existing Codebase & Integration Plan

Based on the Guiding Principles and Target Architecture, the existing codebase components will be utilized as follows:

*   **`recipes.py` (HypothesisManager System):**
    *   **Status:** Keep & Modify.
    *   **Role:** Core symbolic reasoning strategy (GPU-accelerated exhaustive enumerator/validator).
    *   **Modifications:** Wrap its functionality (likely via `TicTacToeAlgorithm` or directly using `HypothesisManager`) within a class conforming to the standard `Strategy` interface. Implement necessary input/output mapping to/from standard formats. Adapt its internal representation to be convertible to/from the standard `CandidateProgram` format. Ensure output labels align with the standard.
*   **`few_shot_algs/sat_hypotheses.py` (SATHypothesesAlgorithm):**
    *   **Status:** Keep & Modify.
    *   **Role:** Core symbolic reasoning strategy (SAT-based hypothesis generator/validator).
    *   **Modifications:** While it already implements `Algorithm`, adapt it to the new standard `Strategy` interface (which might differ slightly). Ensure its output (CNF rules, predictions) is packaged into the standard `CandidateProgram` format, including relevant provenance and confidence.
*   **`tictactoe.py`:**
    *   **Status:** Keep & Modify.
    *   **Role:** Provides the core logic and data generation for the initial TicTacToe test domain.
    *   **Modifications:** Expose functions like `generate_all_answers`, `random_board`, and potentially `tictactoe_no_diags` cleanly for use by the new framework's input generation and verification steps. Ensure label consistency (`label_space`).
*   **`few_shot.py` (Tester Framework):**
    *   **Status:** Discard.
    *   **Reason:** The meta-synthesis framework replaces this classification-focused benchmarking setup. Plotting logic might be reusable later for the Reporting Module.
*   **`few_shot_algs/` (Other Algorithms):**
    *   **Status:** Discard (for MVP). Re-evaluate later.
    *   **Reason:** Most are classification-focused. They could potentially be adapted later as feature extractors or specialized strategies, but are out of scope for the initial MVP.
*   **`few_shot_algs/few_shot_alg.py` (Base Class):**
    *   **Status:** Discard.
    *   **Reason:** A new, more general `Strategy` interface will be defined.
*   **`README.md`, `PROMPT*.txt`, `TODO.txt`:**
    *   **Status:** Archive / Reference Only.
*   **`results/`:**
    *   **Status:** Discard / Archive.

## 5. Initial MVP Scope & Requirements

**5.1. MVP Goal:**
To establish the core orchestration framework and integrate the two primary symbolic reasoning strategies (`recipes.py` system and `SATHypothesesAlgorithm`) identified in the codebase. The MVP will demonstrate the ability to process TicTacToe tasks, generate hypotheses using both strategies, verify them, and report results in a standardized way.

**5.2. MVP Components:**
The MVP will include minimal, functional versions of:
*   Input Orchestrator (JSON Task Format)
*   Basic Feature Extractor Bank (>= 3 simple features)
*   Strategy Dispatcher (Run-all implementation)
*   Strategy Registry (Hardcoded with the two strategies)
*   Strategy Wrappers (for `recipes.py` system and `SATHypothesesAlgorithm`)
*   Standard Program Representation (`CandidateProgram` schema)
*   Basic Verification Engine (I/O matching for TicTacToe)
*   Result Aggregator (Collects and verifies results)
*   Reporting Module (JSON and simple log output)
*   Top-Level Orchestrator Script

**5.3. MVP Requirements:**

*   **[R-MVP-01] Task Definition:** Define and document the standard JSON format for input tasks, including fields for `task_id`, `description` (optional), `inputs` (list of input dicts), `outputs` (list of corresponding output dicts), `constraints` (optional).
*   **[R-MVP-02] Input Parsing:** Implement a module to parse and validate tasks conforming to [R-MVP-01].
*   **[R-MVP-03] Feature Extraction:** Implement at least three basic feature extractors (e.g., input type, input dimensions, value ranges) and a runner module. Define a simple key-value format for extracted features.
*   **[R-MVP-04] Standard Program Representation:** Define a `CandidateProgram` data class/schema containing fields for `program_representation` (string or structured data), `representation_type` (e.g., 'CNF', 'TensorRule'), `source_strategy` (ID), `confidence` (float, optional), `verification_status` (enum), `provenance` (dict).
*   **[R-MVP-05] Standard Strategy Interface:** Define a Python abstract base class or Protocol `Strategy` with methods like `generate(task_spec, features) -> List[CandidateProgram]` and potentially `setup()` / `teardown()`.
*   **[R-MVP-06] Recipes Strategy Wrapper:** Implement a `Strategy` wrapper for the `recipes.py` system. This includes:
    *   Mapping standard task inputs/outputs to the required tensor format.
    *   Invoking the hypothesis generation/validation logic.
    *   Converting resulting valid hypotheses (tensor format) into `CandidateProgram` objects [R-MVP-04].
    *   Handling label space mapping.
*   **[R-MVP-07] SAT Strategy Wrapper:** Implement a `Strategy` wrapper for `SATHypothesesAlgorithm`. This includes:
    *   Ensuring it receives input/outputs correctly.
    *   Invoking its `predict` and `update_history` (or refactored equivalents) logic appropriately within the `generate` call.
    *   Packaging its resulting CNF rules or predictions into `CandidateProgram` objects [R-MVP-04].
*   **[R-MVP-08] Strategy Registry & Dispatcher:** Implement a simple registry holding instances of the two wrapped strategies [R-MVP-06, R-MVP-07] and a dispatcher that iterates through the registry, calling `generate` on each.
*   **[R-MVP-09] Basic Verification:** Implement a `Verifier` module with a function `verify(candidate: CandidateProgram, task_spec) -> VerificationResult`. For the MVP, this function must be able to check the validity of rules/outputs from both strategies against the TicTacToe examples provided in the `task_spec`.
*   **[R-MVP-10] Aggregation & Reporting:** Implement logic to collect `CandidateProgram` lists from the dispatcher, run verification [R-MVP-09] on each, and update their `verification_status`. Implement reporting to output results as a structured JSON file and a simple console log summary.
*   **[R-MVP-11] Orchestration Script:** Create a main script that initializes components and executes the flow: Parse Task -> Extract Features -> Dispatch Strategies -> Aggregate Results -> Verify Candidates -> Report Results.
*   **[R-MVP-12] TicTacToe Test Data:** Create sample task files in the standard JSON format [R-MVP-01] for both standard TicTacToe and the no-diagonal variant, using data generated/validated by `tictactoe.py`.

## 6. Future Work / Roadmap (Post-MVP)

Following the successful implementation of the MVP, future development will focus on expanding the system towards the full architecture:

*   **Strategy Expansion:** Integrate more reasoning strategies (LLMs, ILP, Grammar Induction, etc.).
*   **Enhanced Dispatching:** Implement intelligent strategy selection based on features and task types.
*   **Advanced Features:** Develop more sophisticated feature extractors.
*   **Program Representation:** Implement robust converters between representations.
*   **Equivalence Checking:** Build the layered equivalence checking module.
*   **Advanced Verification:** Integrate property-based testing or other formal methods.
*   **Ranking & Learning:** Develop scoring heuristics and potentially a learned ranking model based on results and feedback.
*   **Robust Caching:** Implement a persistent caching layer.
*   **Provenance Tracking:** Enhance the detail and utility of provenance data.
*   **UI/Dashboard:** Develop a user interface for task submission, monitoring, and result exploration.
*   **Domain Expansion:** Test and adapt the system for new problem domains beyond TicTacToe.
