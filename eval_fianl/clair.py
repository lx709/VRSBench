import json
import logging
import re
import sys
from typing import List, Optional, Tuple

# from grazier import LLMEngine
# _engine_cache = {}

from openai import OpenAI
client = OpenAI()

_CLAIR_PROMPT = """\
You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.
Candidate set:
{candidate_statements}
Reference set:
{target_statements}
On a precise scale from 0 to 100, how likely is it that the candidate set is \
describing the same image as the reference set? (JSON format, with a key "score", \
value between 0 and 100, and a key "reason" with a string value.)
"""

def clair(
    candidates: List[str],
    targets: List[str],
    model: str = "gpt-4o",
    max_tokens: int = 1024,
) -> Tuple[float, Optional[str]]:
    # Compute the CLAIR score for a list of candidates and targets.

    # if model not in _engine_cache:
    #     _engine_cache[model] = LLMEngine.from_string(model)

    # Format the canndidates and targets
    candidate_statements = [f"- {c}\n" for c in candidates]
    target_statements = [f"- {t}\n" for t in targets]
    formatted_prompt = _CLAIR_PROMPT.format(
        candidate_statements="".join(candidate_statements),
        target_statements="".join(target_statements),
    )

    temperature, score, reason = 0.0, None, None
    for _ in range(3):
        # Run the model
        logging.debug(f'CLAIR prompt: "{formatted_prompt}"')
        # response = _engine_cache[model](formatted_prompt, temperature=temperature, max_tokens=max_tokens)[0]

        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-1106",
            model=model, #"gpt-4o-mini"
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": formatted_prompt,
                        },
                    ]
                }
            ],
            max_tokens=100,
        )
        
        response =  response.choices[0].message.content

        logging.debug(f'CLAIR response: "{response.strip()}"')

        # Parse the first JSON object in the response
        try:
            parsed = response.split("{")[1]
            parsed = "{" + parsed.split("}")[0] + "}"
            data = json.loads(parsed)
            score = float(data["score"])
            reason = data.get("reason", 'Unknown')
            break
        except (json.JSONDecodeError, KeyError, IndexError):
            # Try to extract the first number in the response using regex
            parsed = re.findall(r"\d*\.?\d+", response)
            if len(parsed) > 0:
                score = float(parsed[0])
                if score < 1:
                    score *= 100 # This is a weird situation where some models auto-normalize the score for us.

                # Look for the word "reason" in the response, and extract anything after it (ignoring case)
                reason = re.findall(r"(?i)reason.*", response)
                if len(reason) > 0:
                    # Clean up the reason a bit.
                    reason = reason[0].strip()[len('reason'):].replace(':', '').strip()
                else:
                    reason = 'Unknown'
                break
            else:
                logging.warn(
                    f"Could not parse response from CLAIR: {response}. Retrying"
                )
                continue
    else:
        logging.error("Could not parse response from CLAIR after 3 tries. Setting score to 0.")
        score = 0.0
        reason = None

    return score / 100, reason


if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)


    with open(sys.argv[1]) as f:
        data = json.load(f)

    for sample in data:
        score, reason = clair([sample['test']], sample['refs'], model='chat-gpt', max_tokens=128)
        print(f"Score: {score}, Reason: {reason}")