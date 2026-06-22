import os
import json
import numpy as np
import requests
import re
from openai import OpenAI
import time
from typing import Dict, List, Tuple, Optional
from config import DEEPSEEK_API_KEY, BASE_URL, MODEL_NAME

# ================= core_dependencies_the_configuration_area =================

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL
)
MODEL_NAME = MODEL_NAME


class MetaRAGDetector:
    def __init__(
        self,
        model: str = MODEL_NAME,
        verifier_model: str = "gemma3:4b",
        large_verifier_model: str = "deepseek-v4-flash",
        use_local_verifier: bool = True,
        escalation_policy: str = "structure_aware",
        min_anomaly_count: int = 2,
        anomaly_ratio_threshold: float = 0.4,
        use_risk_prior: bool = False,
    ):
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model = model
        self.verifier_model = verifier_model
        self.large_verifier_model = large_verifier_model
        self.use_local_verifier = use_local_verifier

        # RQ2: the_main_rule_is_used_by_default : structure-aware
        self.escalation_policy = escalation_policy
        self.min_anomaly_count = min_anomaly_count
        self.anomaly_ratio_threshold = anomaly_ratio_threshold
        self.use_risk_prior = use_risk_prior

        self.metrics = self._empty_metrics()

    def _empty_metrics(self) -> Dict:
        return {
            "api_prompt_tokens": 0,
            "api_completion_tokens": 0,
            "api_calls": 0,
            "local_prompt_tokens": 0,
            "local_completion_tokens": 0,
            "local_calls": 0,
            "total_latency_sec": 0.0,
            "total_factoids": 0,
            "escalation_count": 0,
            "escalation_reasons": [],
            "factoid_logs": [],
            "stage1_final_score": 0.0,
            "cascade_final_score": 0.0,
            "api_decompose_tokens": 0,
            "api_mutation_tokens": 0,
            "api_stage2_verify_tokens": 0,
            "api_stage1_verify_tokens": 0,
        }

    def _accumulate_api_usage(self, usage, phase: str) -> None:
        if not usage:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = prompt_tokens + completion_tokens

        self.metrics["api_prompt_tokens"] += prompt_tokens
        self.metrics["api_completion_tokens"] += completion_tokens
        self.metrics["api_calls"] += 1

        if phase == "decompose":
            self.metrics["api_decompose_tokens"] += total_tokens
        elif phase == "mutation":
            self.metrics["api_mutation_tokens"] += total_tokens
        elif phase == "stage2_verify":
            self.metrics["api_stage2_verify_tokens"] += total_tokens
        elif phase == "stage1_verify_remote":
            self.metrics["api_stage1_verify_tokens"] += total_tokens

    def _call_llm(self, prompt, system_prompt=None, temperature=0.2, phase: str = "general"):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            self._accumulate_api_usage(response.usage, phase)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def _call_local_ollama(self, user_prompt, system_prompt, temperature: float = 0.0, target_model: Optional[str] = None) -> str:
        chat_url = self.ollama_url
        if "/api/chat" not in chat_url:
            chat_url = "http://localhost:11434/api/chat"

        model_to_use = target_model if target_model else self.verifier_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": model_to_use,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 100,
                "num_ctx": 8192
            }
        }

        try:
            response = requests.post(chat_url, json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                self.metrics["local_prompt_tokens"] += data.get("prompt_eval_count", 0)
                self.metrics["local_completion_tokens"] += data.get("eval_count", 0)
                self.metrics["local_calls"] += 1
                return data.get("message", {}).get("content", "").strip()
            print(f"\n Ollama error (status_code {response.status_code}): {response.text}")
            return ""
        except Exception as e:
            print(f"\n Ollama request_for_collapse: {e}")
            return ""


    def _log(self, message: str = "") -> None:
        if self.verbose:
            print(message)

    def _section(self, title: str, char: str = "=") -> None:
        if not self.verbose:
            return
        line = char * 18
        print(f"\n{line} {title} {line}")

    def _preview(self, text: Optional[str], limit: Optional[int] = None) -> str:
        if text is None:
            return ""
        limit = limit or self.text_preview_chars
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit] + " ..."

    def _format_list(self, items: List[str]) -> str:
        if not items:
            return "[]"
        return "[" + ", ".join(items) + "]"

    def decompose_answer(self, answer: str) -> List[str]:
        system_prompt = """
        You are a fact extraction assistant. 
        Your task is to extract all specific factual propositions from the given text. 
         
        Instructions: 
        1. Extract every distinct factual statement present in the input, even if the statement is incorrect, ambiguous, or nonsensical. 
        2. Each extracted proposition must be a complete, standalone sentence. 
        3. Each sentence must express only one atomic fact. (An atomic fact cannot be split into simpler factual statements.) 
        4. If a sentence contains multiple facts, split them into multiple atomic fact sentences. 
        5. Do not paraphrase, rewrite, summarize, interpret, infer, or judge any part of the input. Only extract and restate what is explicitly written. 
        6. Do not omit or correct any statements, regardless of their factual accuracy. 
        7. Output your answer as a JSON array of strings, with each element being one atomic factual sentence.  
        8. **CRITICAL: Resolve all pronouns (he, she, it, they, this) to their exact entity names based on the context.** Do not use pronouns in the extracted facts.
        Example: 
        Input: Marie Curie discovered polonium and radium, and Albert Einstein developed the theory of relativity in 1905.  
        Output: [  "Marie Curie discovered polonium.",
                    "Marie Curie discovered radium.",
                     "Albert Einstein developed the theory of relativity in 1905." 
                 ]  
        Now, extract atomic facts from this text: 
        """
        prompt = f"Input:\n{answer}"
        response = self._call_llm(prompt, system_prompt, temperature=0.0, phase="decompose")
        try:
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            factoids = json.loads(cleaned_response)
            return factoids if isinstance(factoids, list) else [answer]
        except json.JSONDecodeError:

            return [answer]

    def generate_mutations(self, question: str, factoid: str, n: int = 2) -> Tuple[List[str], List[str]]:
        syn_prompt = f"""
        You will be given a question and a factual answer (factoid).  
        Your task is to generate {n} *synonyms* (paraphrased statements with the same meaning) of the factoid, 
        based on the context of the question.  
        Instructions: 
        - Each output must be a single atomic factual claim (cannot be split into smaller facts). 
        - Use only information explicitly present in the question or factoid. Do not invent, infer, or add external knowledge. 
        - A correct synonym is a statement that means exactly the same thing as the factoid, even if the wording is different. 
        - Do not output partial phrases, keywords, or combine/split facts. 
        - Each synonym must be a complete, grammatically correct sentence. 
        - Just return the sentences, one per line, without numbers, bullets, or any other output.

        Example: Question: Where was Einstein born? 
        Factoid: Einstein was born in Germany. Good Synonym: Germany is the country where Einstein was born.
        Bad Synonym: Einstein visited Germany. (not equivalent) 
        Bad Synonym: Einstein was born. (incomplete)

        """
        prompt = f"Input:\nQuestion: {question}\nFactoid: {factoid}"
        synonyms = self._call_llm(prompt, syn_prompt, temperature=0.2, phase="mutation").split('\n')
        synonyms = [s.strip() for s in synonyms if s.strip()][:n]

        ant_prompt = f"""
        You will be given a question and a factual answer (factoid).
        Your task is to generate {n} negations (contradictory statements) of the factoid, based on the context of the question.
        Instructions:
        - Each negation must directly contradict the factoid, focusing on what the question asks.
        - Do not add new information not present in the factoid or question.
        - Do not use double negations or wording that preserves the original meaning.
        - Each negation must be a meaningful, grammatically correct sentence.
        - Do not introduce unrelated facts.
        - Ensure that each negation is relevant to the question's context.
        - **Just return the sentences, one per line, without numbers or bullets, and nothing else.**

        Example: 
        Question: 
        Where was Einstein born? 
        Factoid: Einstein was born in Germany. 
        Good Antonym: Einstein was not born in Germany. 
        Bad Antonym: Einstein visited Germany. (not a contradiction) 
        Bad Antonym: Einstein was born in Austria. (adds new information) 
        Bad Antonym: Einstein was not not born in Germany. (double negation) 
        Bad Antonym: Was not born in Germany. (missing subject)

        """
        antonyms = self._call_llm(prompt, ant_prompt, temperature=0.2, phase="mutation").split('\n')
        antonyms = [a.strip() for a in antonyms if a.strip()][:n]
        return synonyms, antonyms

    def is_high_risk_factoid(self, factoid_text: str) -> Optional[str]:
        # if not self.use_risk_prior:
        #     return None
        # text = str(factoid_text).lower()
        # if re.search(r'\b\d+\b', text):
        #     return "RiskPrior: Numerical/Temporal"
        # if re.search(r'\b(not|never|no|none|cannot|without|lack)\b', text):
        #     return "RiskPrior: Negation"
        # if re.search(r'\b(more|less|most|least|better|worse|highest|lowest|all|every|only|always)\b', text):
        #     return "RiskPrior: Scope/Comparison"
        return None

    def verify_statement(self, statement: str, context: str, escalate: bool = False) -> str:
        sys_prompt = f"""
        You will be given a statement and passages that represent the ground truth.  
        Determine if the statement is supported by the passage, either explicitly or through clear implication.  

        Answer with one of the following **only**: 
        -YES: if the statement is clearly and completely supported by the passages. 
        -NO: if the statement is contradicted or directly refuted by the passages. 
        -NOT SURE: if the passage does not contain enough information to confirm or deny the statement.  

        Respond with YES, NO, or NOT SURE. Then, in one short sentence, explain the reason for your answer.  

        Examples:  

        Passages (Ground Truth): "Alice was born in Paris and moved to New York at the age of five." 
        Statement: "Alice spent her early childhood in France." 
        Answer: YES. The passage states Alice was born in Paris, which is in France.  

        Passages (Ground Truth): "Bob has never visited Japan but plans to travel there next summer." 
        Statement: "Bob visited Japan last year." 
        Answer: NO. The passage says Bob has never visited Japan.

        Passages (Ground Truth): "Carol enjoys outdoor activities like hiking and cycling." 
        Statement: "Carol loves swimming." 
        Answer: NOT SURE. There is no information in the passages about Carol and swimming.  

        """

        prompt = f"""
        Now, perform the task:  
        Passages (Ground Truth): ${context} 
        Statement: ${statement} 
        Answer:
        """

        if self.use_local_verifier:
            if escalate:
                response = self._call_llm(prompt, sys_prompt, temperature=0.0, phase="stage2_verify")
            else:
                response = self._call_local_ollama(prompt, sys_prompt, temperature=0.0, target_model=self.verifier_model)
        else:
            phase = "stage2_verify" if escalate else "stage1_verify_remote"
            response = self._call_llm(prompt, sys_prompt, temperature=0.0, phase=phase)

        upper_res = response.upper()
        if "NOT SURE" in upper_res:
            return "NOT SURE"
        match = re.search(r'\b(YES|NO)\b', upper_res)
        if match:
            return match.group(1)
        return "NOT SURE"

    @staticmethod
    def calculate_score(verification_result: str, mutation_type: str) -> float:
        if mutation_type == "synonym":
            return {"NO": 1.0, "NOT SURE": 0.5, "YES": 0.0}.get(verification_result, 0.5)
        if mutation_type == "antonym":
            return {"YES": 1.0, "NOT SURE": 0.5, "NO": 0.0}.get(verification_result, 0.5)
        return 0.0

    def _compute_factoid_score(self, syn_decisions: List[str], ant_decisions: List[str]) -> float:
        penalties = []
        for dec in syn_decisions:
            penalties.append(self.calculate_score(dec, "synonym"))
        for dec in ant_decisions:
            penalties.append(self.calculate_score(dec, "antonym"))
        return float(np.mean(penalties)) if penalties else 0.0

    def _decide_escalation(self, fact: str, syn_decisions: List[str], ant_decisions: List[str]) -> Tuple[bool, List[str], Dict]:
        reasons: List[str] = []
        syn_total = len(syn_decisions) if syn_decisions else 1
        ant_total = len(ant_decisions) if ant_decisions else 1

        syn_anomaly_count = sum(d in {"NO", "NOT SURE"} for d in syn_decisions)
        ant_anomaly_count = sum(d in {"YES", "NOT SURE"} for d in ant_decisions)
        syn_anomaly_ratio = syn_anomaly_count / syn_total
        ant_anomaly_ratio = ant_anomaly_count / ant_total

        both_yes = ("YES" in syn_decisions and "YES" in ant_decisions)
        both_no = ("NO" in syn_decisions and "NO" in ant_decisions)
        logical_paradox = both_yes or both_no

        risk_reason = self.is_high_risk_factoid(fact)

        if self.escalation_policy == "strict":
            if syn_anomaly_count > 0:
                reasons.append("Strict: Synonym anomaly")
            if ant_anomaly_count > 0:
                reasons.append("Strict: Antonym anomaly")
            if logical_paradox:
                reasons.append("Strict: Logical paradox")
            if risk_reason:
                reasons.append(risk_reason)

        elif self.escalation_policy == "paradox_only":
            if logical_paradox:
                reasons.append("ParadoxOnly: Logical paradox")

        else:  # structure_aware
            if logical_paradox:
                reasons.append("StructureAware: Logical paradox")
            if syn_anomaly_count >= self.min_anomaly_count or syn_anomaly_ratio >= self.anomaly_ratio_threshold:
                reasons.append(
                    f"StructureAware: Synonym anomaly threshold reached ({syn_anomaly_count}/{syn_total})"
                )
            if ant_anomaly_count >= self.min_anomaly_count or ant_anomaly_ratio >= self.anomaly_ratio_threshold:
                reasons.append(
                    f"StructureAware: Antonym anomaly threshold reached ({ant_anomaly_count}/{ant_total})"
                )

            if risk_reason and reasons:
                reasons.append(risk_reason)

        if reasons:
            print(" triggering_the_upgrade_for_the_following_reasons:")
            for idx, reason in enumerate(reasons, start=1):
                print(f"   {idx}. {reason}")
        else:
            print(" No upgrade is triggered; the results from Stage 1 are carried over.")

        stats = {
            "syn_anomaly_count": syn_anomaly_count,
            "ant_anomaly_count": ant_anomaly_count,
            "syn_anomaly_ratio": round(syn_anomaly_ratio, 4),
            "ant_anomaly_ratio": round(ant_anomaly_ratio, 4),
            "logical_paradox": logical_paradox,
            "both_yes": both_yes,
            "both_no": both_no,
            "risk_prior": risk_reason or ""
        }
        return (len(reasons) > 0), reasons, stats

    def detect(self, query: str, response: str, retrieved_context: str, n_mutations: int = 2):
        self.metrics = self._empty_metrics()
        start_time = time.time()

        print(f"--- let_s_start_the_analysis ---\nQuery: {query}\nResponse: {response}\n")

        factoids = self.decompose_answer(response)
        self.metrics["total_factoids"] = len(factoids)
        print(f" extracting_the_fact_of_the_atom (Factoids): {factoids}")


        stage1_factoid_scores: List[float] = []
        cascade_factoid_scores: List[float] = []

        for f_idx, fact in enumerate(factoids):
            print(f"\n   facts_are_being_dealt_with [{f_idx + 1}]: {fact}")
            syns, ants = self.generate_mutations(query, fact, n=n_mutations)


            print("     (Synonyms):")
            for i, s in enumerate(syns):
                print(f"       {i + 1}. {s}")

            print("     (Antonyms):")
            for i, a in enumerate(ants):
                print(f"       {i + 1}. {a}")


            syn_stage1 = [self.verify_statement(s, retrieved_context, escalate=False) for s in syns]
            ant_stage1 = [self.verify_statement(a, retrieved_context, escalate=False) for a in ants]
            stage1_score = self._compute_factoid_score(syn_stage1, ant_stage1)
            stage1_factoid_scores.append(stage1_score)

            print(f" Stage 1 decisions | Syn={self._format_list(syn_stage1)} | Ant={self._format_list(ant_stage1)}")
            print(f" Stage 1 factoid score = {stage1_score:.4f}")

            escalate_flag = False
            trigger_reasons: List[str] = []
            trigger_stats: Dict = {}
            syn_final, ant_final = syn_stage1[:], ant_stage1[:]
            stage2_score = stage1_score

            if self.use_local_verifier:
                escalate_flag, trigger_reasons, trigger_stats = self._decide_escalation(fact, syn_stage1, ant_stage1)
                if escalate_flag:
                    self.metrics["escalation_count"] += 1
                    self.metrics["escalation_reasons"].extend(trigger_reasons)

                    print("\n [Stage 2] Triggering upgrades and reusing large models for verification mutation...")

                    syn_final = []
                    for idx, s in enumerate(syns, start=1):
                        print(f"   -> verify_again Syn {idx}")
                        syn_final.append(self.verify_statement(s, retrieved_context, escalate=True))
                    ant_final = []
                    for idx, a in enumerate(ants, start=1):
                        print(f"   -> verify_again Ant {idx}")
                        ant_final.append(self.verify_statement(a, retrieved_context, escalate=True))

                    stage2_score = self._compute_factoid_score(syn_final, ant_final)
                    print(f" Stage 2 decisions | Syn={self._format_list(syn_final)} | Ant={self._format_list(ant_final)}")
                    print(f" Cascade factoid score = {stage2_score:.4f}")
                    print(f" Score change = {stage1_score:.4f} -> {stage2_score:.4f}")

            cascade_factoid_scores.append(stage2_score)

            self.metrics["factoid_logs"].append({
                "factoid_id": f_idx + 1,
                "factoid_text": fact,
                "synonyms": syns,
                "antonyms": ants,
                "stage1_syn_decisions": syn_stage1,
                "stage1_ant_decisions": ant_stage1,
                "final_syn_decisions": syn_final,
                "final_ant_decisions": ant_final,
                "stage1_factoid_score": round(stage1_score, 4),
                "cascade_factoid_score": round(stage2_score, 4),
                "escalated": escalate_flag,
                "trigger_reasons": trigger_reasons,
                **trigger_stats,
            })

        self.metrics["stage1_final_score"] = max(stage1_factoid_scores) if stage1_factoid_scores else 0.0
        self.metrics["cascade_final_score"] = max(cascade_factoid_scores) if cascade_factoid_scores else 0.0
        self.metrics["total_latency_sec"] = time.time() - start_time

        return self.metrics["cascade_final_score"], self.metrics
