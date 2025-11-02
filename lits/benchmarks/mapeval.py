import re
def construct_prompt(item):
    """ 
    return: 
        Example: I am currently staying at Hostel El Grial in Cusco, Peru. I want to visit Sacsayhuamán for 1.5 hours, \
            Qorikancha Temple for 1 hour, the Cusco Cathedral for 1.5 hours, and the Mercado Central de San Pedro for 1 hour. \
            have 6 hours available. I will leave my hostel at 8 am. Give the most optimized order to visit these places. \
            I will drive my car.\
            Choose the answer from the following options (1/2/3/4). So, the output format will be "^^Option_Number^^". Choose the correct answer from the following options: \
                Option1: Saqsaywaman -> Cusco Cathedral -> Mercado Central de San Pedro -> Qorikancha (35 mins), \
                Option2: Cusco Cathedral -> Qorikancha -> Mercado Central de San Pedro -> Saqsaywaman (37 mins), \
                Option3: Saqsaywaman -> Qorikancha -> Mercado Central de San Pedro -> Visit Cusco Cathedral (48 mins), \
                Option4: Mercado Central de San Pedro -> Qorikancha -> Cusco Cathedral -> Saqsaywaman (33 mins), 
    """
    prompt = (
        item["question"]
        + "Choose the answer from the following options (1/2/3/4). So, the output format will be \"^^Option_Number^^\". Choose the correct answer from the following options: "
    )
    
    if item["classification"] is None:
        prompt = prompt + "Option0: Unanswerable, "
        
    for i in range(len(item["answer"]["options"])):
        if(item["answer"]["options"][i] == ""):
            break
        prompt = (
            prompt
            + "Option"
            + str(i + 1)
            + ": "
            + item["answer"]["options"][i]
            + ", "
        )
    # print("Prompt is created. Now passing to the model.", prompt)
    return prompt

def extract(s):
    for char in s:
        if char.isdigit():
            return char
    return None  # Return None if no numeric character is found

def retrieve_answer(text: str) -> int | None:
    """
    Extracts the option number from a string formatted as ^^Option_Number^^ 
    or similar variants (^^2^^, ^^Option2^^).
    """
    match = re.search(r"\^\^(?:Option_?|)(\d+)\^\^", text.strip())
    return int(match.group(1)) if match else None


def make_answer_extractor(primary_extractor, fallback_fn):
    """Compose a tag-based extractor with an additional fallback heuristic."""
    def extractor(text: str) -> list[str]:
        answers = [ans.strip() for ans in primary_extractor(text) if ans.strip()]
        if answers:
            return answers
        fallback = fallback_fn(text)
        if fallback is None:
            return []
        return [str(fallback)]

    return extractor