# app/llm.py
import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key="skAxEQ216k86ivIm-6F3QsmaA4ybwMEA8K3FYY90BlL3i6T3BlbkFJmi_G5oiL1Ugg21PrajbyYRSSeRtCgA")


def explain_negative(review_text: str) -> str:
    """
    Uses OpenAI to return a short, specific explanation for why the review is negative.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-3.5-turbo if you want cheaper/faster
            messages=[{
                "role": "user",
                "content": (
                    "Explain in one short sentence why the following customer review is negative. "
                    "Be specific and objective.\n\n"
                    f"Review: {review_text}"
                )
            }],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def rephrase_brand_friendly(review_text: str) -> str:
    """
    Uses OpenAI to rewrite the review in a polite, neutral, brand-friendly tone without changing meaning.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite the following customer review in a polite, neutral, and brand-friendly tone "
                    "without changing its meaning. Avoid emojis or slang.\n\n"
                    f"Review: {review_text}"
                )
            }],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating rephrased review: {str(e)}"
