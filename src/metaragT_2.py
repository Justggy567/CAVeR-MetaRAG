import os
import json
import numpy as np
import requests
import re
from openai import OpenAI
import time
from config import DEEPSEEK_API_KEY, BASE_URL, MODEL_NAME


# ================= core_dependencies_the_configuration_area =================



client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL
)
MODEL_NAME = MODEL_NAME



class MetaRAGDetector:
    def __init__(self, model=MODEL_NAME, verifier_model="llama3", use_local_verifier=True, verifier_options=None):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = model
        self.verifier_model = verifier_model
        self.use_local_verifier = use_local_verifier
        self.verifier_options = verifier_options or {}

        self.metrics = {
            "api_prompt_tokens": 0, "api_completion_tokens": 0, "api_calls": 0,
            "local_prompt_tokens": 0, "local_completion_tokens": 0, "local_calls": 0,
            "total_latency_sec": 0.0
        }

    def _call_llm(self, prompt, system_prompt=None, temperature=0.2):
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

            if response.usage:
                self.metrics["api_prompt_tokens"] += response.usage.prompt_tokens
                self.metrics["api_completion_tokens"] += response.usage.completion_tokens
            self.metrics["api_calls"] += 1

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def _call_local_ollama(self, user_prompt, system_prompt, temperature=0.0):
        url = "http://localhost:11434/api/chat"


        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        payload = {
            "model": self.verifier_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.1,
                "num_predict": 100
            }
        }


        think_flag = self.verifier_options.get("think")
        if think_flag is not None:
            payload["think"] = think_flag


        extra_options = self.verifier_options.get("options", {})
        if extra_options:
            payload["options"].update(extra_options)

        try:

            print(f"Ollama {self.verifier_model}")
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()


                self.metrics["local_prompt_tokens"] += data.get("prompt_eval_count", 0)
                self.metrics["local_completion_tokens"] += data.get("eval_count", 0)
                self.metrics["local_calls"] += 1


                content = data.get("message", {}).get("content", "")
                return content.strip()

            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return "NOT SURE"
        except Exception as e:
            print(f"Ollama connection_failed: {e}")
            return "NOT SURE"

    def decompose_answer(self, answer):

        system_prompt = f"""
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
        response = self._call_llm(prompt, system_prompt, temperature=0.0)

        try:
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            factoids = json.loads(cleaned_response)
            return factoids
        except json.JSONDecodeError:
            print(f"parsing_failed_original_output: {response}")
            return [answer]

    def generate_mutations(self, question, factoid, n=2):
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
        prompt = f"Input:\n Question: {question}\nFactoid: {factoid}"

        synonyms = self._call_llm(prompt, syn_prompt,temperature=0.2).split('\n')
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

        antonyms = self._call_llm(prompt, ant_prompt,temperature=0.2).split('\n')
        antonyms = [a.strip() for a in antonyms if a.strip()][:n]

        return synonyms, antonyms

    def verify_statement(self, statement, context):
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
            response = self._call_local_ollama(prompt,sys_prompt, temperature=0.0)
        else:

            response = self._call_llm(prompt, sys_prompt,temperature=0.0)




        print(f"[Verifier raw_output]: {response}")

        upper_res = response.upper()


        if "NOT SURE" in upper_res:
            return "NOT SURE"



        match = re.search(r'\b(YES|NO)\b', upper_res)

        if match:
            extracted_answer = match.group(1)
            return extracted_answer


        return "NOT SURE"



    def calculate_score(self, verification_results, mutation_type):
        penalty = 0.0
        label = verification_results

        if mutation_type == "synonym":
            if label == "NO":
                penalty = 1.0
            elif label == "NOT SURE":
                penalty = 0.5
            elif label == "YES":
                penalty = 0.0
        elif mutation_type == "antonym":
            if label == "YES":
                penalty = 1.0
            elif label == "NOT SURE":
                penalty = 0.5
            elif label == "NO":
                penalty = 0.0

        return penalty


    def detect(self, query, response, retrieved_context, n_mutations=2):


        self.metrics = {k: 0 for k in self.metrics}
        start_time = time.time()

        print(f"--- let_s_start_the_analysis ---\nQuery: {query}\nResponse: {response}\n")


        factoids = self.decompose_answer(response)
        print(f"1. (Factoids): {factoids}")

        factoid_scores = []

        for f_idx, fact in enumerate(factoids):
            print(f"\n   facts_are_being_dealt_with [{f_idx + 1}]: {fact}")


            syns, ants = self.generate_mutations(query, fact, n=n_mutations)

            penalties = []


            for s in syns:
                decision = self.verify_statement(s, retrieved_context)
                p = self.calculate_score(decision, "synonym")
                penalties.append(p)
                print(f"     [synonyms_and_variations] '{s}' -> judgment: {decision} | punished: {p}")


            for a in ants:
                decision = self.verify_statement(a, retrieved_context)
                p = self.calculate_score(decision, "antonym")
                penalties.append(p)
                print(f"     [the_antonym_has_changed] '{a}' -> judgment: {decision} | punished: {p}")


            avg_score = np.mean(penalties) if penalties else 0
            factoid_scores.append(avg_score)
            print(f"   => factoid [{f_idx + 1}] illusions_are_separated: {avg_score:.2f}")


        final_score = max(factoid_scores) if factoid_scores else 0

        self.metrics["total_latency_sec"] = time.time() - start_time
        print(f"\n=== final_hallucination_score (Max): {final_score:.2f} ===")
        if final_score >= 0.5:
            print("conclusion_high_risk_hallucinations_detected")
        else:
            print("conclusion_the_answer_is_reliable")

        return final_score, self.metrics





if __name__ == "__main__":

    VERIFIER_MODELS = {
        "deepseek-v4-flash": {"model": "deepseek-v4-flash", "is_local": False},

        "Gemma3-4B": {"model": "gemma3:4b", "is_local": True},

        "Phi4-14B": {"model": "phi4", "is_local": True},

        "Gemma3-12B": {"model": "gemma3:12b", "is_local": True},
    }

