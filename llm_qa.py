from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List, Dict, Any

class LLMQA:
    def __init__(self, model_name: str = 'google/flan-t5-base') -> None:
        print(f"Loading LLM model: {model_name}")

        device = 0 if torch.cuda.is_available() else -1
        device_name = 'GPU' if device == 0 else 'CPU'

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            self.pipe = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                device=device
            )

            self.prompt_template = """Based on the following context, answer the question. If the answer is not in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

            print(f"LLM loaded on {device_name}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        context_text = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['content'][:500]}"
            for chunk in context_chunks[:3]
        ])

        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )

        try:
            result = self.pipe(prompt)[0]['generated_text']
            answer = result.strip()

        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = "Sorry, I encountered an error generating the answer."

        return answer

    def generate_answer_with_citations(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_chunks = [result['chunk'] for result in search_results]

        answer = self.generate_answer(query, context_chunks)

        citations = []
        for i, result in enumerate(search_results[:3]):
            chunk = result['chunk']
            citations.append({
                'rank': i + 1,
                'source': chunk['source'],
                'page': chunk['page'],
                'type': chunk['type'],
                'relevance_score': result['score']
            })

        return {
            'answer': answer,
            'citations': citations,
            'context_used': len(context_chunks)
        }


class SimpleQA:
    def __init__(self) -> None:
        print("SimpleQA initialized (no LLM)")

    def generate_answer_with_citations(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not search_results:
            return {
                'answer': "No relevant information found in the document.",
                'citations': [],
                'context_used': 0
            }
        top_chunks = search_results[:3]

        answer_parts = []
        for result in top_chunks:
            chunk = result['chunk']
            snippet = chunk['content'][:200].strip()
            if snippet:
                answer_parts.append(f"From {chunk['source']}: {snippet}...")

        answer = "\n\n".join(answer_parts) if answer_parts else "No relevant information found."

        citations = []
        for i, result in enumerate(top_chunks):
            chunk = result['chunk']
            citations.append({
                'rank': i + 1,
                'source': chunk['source'],
                'page': chunk['page'],
                'type': chunk['type'],
                'relevance_score': result['score']
            })

        return {
            'answer': answer,
            'citations': citations,
            'context_used': len(search_results)
        }


if __name__ == "__main__":
    test_results = [
        {
            'chunk': {
                'content': 'Qatar economy grew by 5% in 2024 driven by strong non-hydrocarbon sector growth.',
                'page': 1,
                'type': 'text',
                'source': 'Page 1'
            },
            'score': 0.85
        },
        {
            'chunk': {
                'content': 'The banking sector remains healthy with strong capital ratios.',
                'page': 2,
                'type': 'text',
                'source': 'Page 2'
            },
            'score': 0.72
        }
    ]

    try:
        print("\n1. Testing LLMQA...")
        qa = LLMQA()
        result = qa.generate_answer_with_citations("What is Qatar's growth?", test_results)
        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {len(result['citations'])} sources")

    except Exception as e:
        print(f"\nLLMQA failed: {e}")
        print("\n2. Testing SimpleQA fallback...")
        qa = SimpleQA()
        result = qa.generate_answer_with_citations("What is Qatar's growth?", test_results)
        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {len(result['citations'])} sources")
