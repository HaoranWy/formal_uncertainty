"""
Prompts for Autoformalization (Natural Language -> SMT-LIB v2).
Based on the methodology of 'Grammars of Formal Uncertainty'.

Categories:
1. SYMBOLIC_LOGIC_PROMPT: For ProofWriter, FOLIO, ProntoQA.
   Uses UFLIA (Uninterpreted Functions + Linear Integer Arithmetic).
   Focuses on quantifiers (forall, exists) and logical implication.
   Uses Proof by Contradiction (assert (not hypothesis)).

2. COMMONSENSE_ARITHMETIC_PROMPT: For StrategyQA.
   Uses QF_LIA (Quantifier-Free Linear Integer Arithmetic) or LIA.
   Focuses on mapping facts to integer constraints (years, counts).
"""

from typing import Optional

# ==============================================================================
# 1. System Prompts
# ==============================================================================

SYSTEM_PROMPT_LOGIC = """You are an expert in formal logic and automated reasoning using Z3.
Your task is to translate natural language premises and a hypothesis into a valid SMT-LIB v2 script.

**Instructions:**
1. Use the logic `(set-logic UFLIA)` or `(set-logic ALL)`.
2. Declare a sort `Entity` for objects if necessary.
3. Translate all Context/Premises into `(assert ...)` statements.
4. **Crucial**: To check if the Hypothesis is True, use **Proof by Contradiction**.
   - You must ASSERT THE NEGATION of the Hypothesis.
   - Example: If Hypothesis is "Bob is green", write `(assert (not (IsGreen Bob)))`.
5. End the script with `(check-sat)`.
6. Do not wrap the code in Markdown blocks (like ```smt), just provide the plain code if possible, or minimal wrapping.
"""

SYSTEM_PROMPT_ARITHMETIC = """You are an expert in commonsense reasoning and SMT-LIB v2 programming.
Your task is to model real-world facts using Linear Integer Arithmetic (LIA) to verify a claim.

**Instructions:**
1. Use the logic `(set-logic QF_LIA)` or `(set-logic LIA)`.
2. Represent entities, dates, or quantities as Integer constants/variables.
3. Add assertions to represent the known facts (World Knowledge).
4. Add assertions to represent the logic of the question.
5. To verify the answer, assert the condition that would make the answer "False" (Refutation) OR "True" depending on the phrasing, but ensure the logic is self-contained.
6. End with `(check-sat)`.
"""

# ==============================================================================
# 2. Few-Shot Examples (User Prompts)
# ==============================================================================

EXAMPLES_LOGIC = """
---
Example 1:
Context: "Bob is green. If someone is green, they are happy."
Hypothesis: "Bob is happy."

;; SMT-LIB Output:
(set-logic UFLIA)
(declare-sort Entity 0)
(declare-fun Green (Entity) Bool)
(declare-fun Happy (Entity) Bool)
(declare-const Bob Entity)

; Premise: Bob is green
(assert (Green Bob))

; Premise: If someone is green, they are happy
(assert (forall ((x Entity)) (=> (Green x) (Happy x))))

; Hypothesis Check: We assert the NEGATION of "Bob is happy".
; If result is UNSAT, the hypothesis is Proven True.
(assert (not (Happy Bob)))

(check-sat)
---

Example 2:
Context: "All bears are big. Some bears are white. Dave is a bear."
Hypothesis: "Dave is big."

;; SMT-LIB Output:
(set-logic UFLIA)
(declare-sort Entity 0)
(declare-fun Bear (Entity) Bool)
(declare-fun Big (Entity) Bool)
(declare-fun White (Entity) Bool)
(declare-const Dave Entity)

; Premise: All bears are big
(assert (forall ((x Entity)) (=> (Bear x) (Big x))))

; Premise: Some bears are white
(assert (exists ((x Entity)) (and (Bear x) (White x))))

; Premise: Dave is a bear
(assert (Bear Dave))

; Hypothesis Check: Negate "Dave is big"
(assert (not (Big Dave)))

(check-sat)
---
"""

EXAMPLES_ARITHMETIC = """
---
Example 1:
Question: "Did Aristotle use a laptop?"

;; SMT-LIB Output:
(set-logic QF_LIA)
(declare-const Year_Aristotle_Death Int)
(declare-const Year_Laptop_Invention Int)

; World Knowledge
(assert (= Year_Aristotle_Death (- 322))) ; 322 BC
(assert (= Year_Laptop_Invention 1981))

; Verification Logic:
; To use a laptop, one must be alive after it was invented.
; We assert the scenario where Aristotle DID use a laptop to see if it conflicts with reality.
(assert (>= Year_Aristotle_Death Year_Laptop_Invention))

(check-sat)
---

Example 2:
Question: "Are there more wheels on a car than legs on a cat?"

;; SMT-LIB Output:
(set-logic QF_LIA)
(declare-const Wheels_Car Int)
(declare-const Legs_Cat Int)

; World Knowledge
(assert (= Wheels_Car 4))
(assert (= Legs_Cat 4))

; Verification Logic:
; Claim: Wheels_Car > Legs_Cat.
; We negate the claim to check for contradiction.
(assert (not (> Wheels_Car Legs_Cat)))

(check-sat)
---
"""

# ==============================================================================
# 3. Prompt Builder Class
# ==============================================================================

class PromptGenerator:
    def __init__(self):
        self.logic_datasets = ["proofwriter", "folio", "prontoqa"]
        self.arithmetic_datasets = ["strategyqa"]

    def _construct_message(self, system_content: str, user_content: str) -> list:
        """
        Constructs the message format for OpenAI/ChatCompletion APIs.
        """
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def get_prompt(self, dataset_name: str, question: str, context: Optional[str] = None, use_cot: bool = False) -> list:
        """
        Generates the full prompt messages list for the API.
        
        Args:
            dataset_name: Name of the dataset (lowercase).
            question: The target question or hypothesis.
            context: The context or premises (required for Logic tasks).
            use_cot: Whether to append 'Think step by step' instructions (for models like DeepSeek-R1).
        
        Returns:
            A list of message dictionaries [{'role': '...', 'content': '...'}, ...]
        """
        dataset_key = dataset_name.lower().strip()
        
        # 1. Select Template Strategy
        if dataset_key in self.arithmetic_datasets:
            system_prompt = SYSTEM_PROMPT_ARITHMETIC
            few_shot = EXAMPLES_ARITHMETIC
            # StrategyQA通常没有单独的context字段，context隐含在常识中
            user_input = f"Question: \"{question}\"\n\n;; SMT-LIB Output:\n"
        else:
            # Default to Logic (ProofWriter, etc)
            system_prompt = SYSTEM_PROMPT_LOGIC
            few_shot = EXAMPLES_LOGIC
            # 构造输入
            ctx_str = f'Context: "{context}"\n' if context else ""
            user_input = f"{ctx_str}Hypothesis: \"{question}\"\n\n;; SMT-LIB Output:\n"

        # 2. Append CoT Instruction if needed
        if use_cot:
            user_input = "Please think step by step to analyze the logic first.\n" + user_input

        # 3. Assemble the final content
        # Combine Few-shot examples with the new Task
        full_user_content = f"{few_shot}\n--- NEW TASK ---\n{user_input}"

        return self._construct_message(system_prompt, full_user_content)

# ==============================================================================
# 4. Quick Test Block
# ==============================================================================
if __name__ == "__main__":
    generator = PromptGenerator()
    
    # Test 1: ProofWriter
    prompt_logic = generator.get_prompt(
        "proofwriter", 
        question="Is the cat nice?", 
        context="The cat eats mice. If something eats mice, it is nice."
    )
    print("=== Logic Prompt User Content (Snippet) ===")
    print(prompt_logic[1]['content'][-300:]) # 打印最后300字符查看

    # Test 2: StrategyQA
    prompt_arith = generator.get_prompt(
        "strategyqa", 
        question="Can a sound barrier be broken by a car?"
    )
    print("\n=== Arithmetic Prompt User Content (Snippet) ===")
    print(prompt_arith[1]['content'][-300:])