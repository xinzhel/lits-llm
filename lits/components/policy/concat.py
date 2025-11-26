from ..base import Policy
import logging
import re
from ...lm.base import HfChatModel
from ...structures import State, Action
from ..utils import verbalize_concat_state, extract_existing_steps, create_role

logger = logging.getLogger(__name__)

class ConcatPolicy(Policy):
    """Policy that generates reasoning actions by concatenating new steps to the existing trace."""
    
    def __init__(self, **kwargs):
        self.check_action_sim = kwargs.pop('check_action_sim', False)
        super().__init__(**kwargs)

    def _generate_msg(self, query, state: State, critic: str = None, at_depth_limit=False) -> str:

        user_message = verbalize_concat_state(query, state)
        if critic:
            user_message += "Advice: " + critic + "\n"
        
        user_message += "Step " + str(len(state) + 1) + ": "
        if at_depth_limit:
            user_message += "This is the last step, and the answer to the question has to be reached. "
        self.base_model.sys_prompt = self.task_prompt_spec
        return user_message

    def _check_sim(self, embedding, exist_embeddings):
        for j, exist_embedding in enumerate(exist_embeddings):
            similarity = (embedding * exist_embedding).sum(dim=-1)
            
            if similarity > 0.98: # 0.95 is the threshold for similarity
                return True, j, similarity
            
        return False, -1, 0

    def _get_actions(self, query, state: State,  n_actions, temperature, at_depth_limit, query_idx, critic: str=None, from_phase="") -> list[Action]:
        """ 
        Args:
            at_depth_limit: This is for RAP, not REST
        """
        assert isinstance(query_idx, int), f"example_idx should be an integer, got {query}"
        # print(f"example_idx: {example_idx}")
        if isinstance(self.base_model, HfChatModel):
            txt_or_msg = self._generate_msg(query, state, critic=critic, at_depth_limit=at_depth_limit)
        else:
            txt_or_msg = self.task_prompt_spec + self._generate_msg(query, state, critic=critic, at_depth_limit=at_depth_limit)

        outputs = []
        embeddings = []
        existing_steps = extract_existing_steps(state)
        for idx in range(0, n_actions):
            n_retry_repeat = 0
            while True:
                if self.check_action_sim:
                    assert len(embeddings) == len(outputs), "embeddings and outputs should have the same length"
                    # print(txt_or_msg    )
                    output_text, embedding = self.base_model(txt_or_msg, role=create_role("policy", query_idx, from_phase), temperature=temperature, \
                        max_length=self.max_length, max_new_tokens=self.max_new_tokens, top_p=self.top_p, \
                        new_sent_stop=False, enable_thinking=False, return_embedding=self.check_action_sim)
                    # print(output_text.text.strip())
                    # print("embedding:", embedding)
                    output_text = output_text.text.strip() # enable_thinking=False to make generated action stop correctly using new_sent_stop
                    high_sim, high_sim_idx, similarity = self._check_sim(embedding, embeddings)
                    if high_sim:
                        logger.debug(f"!!!!!!!!!! Found similar embedding (Begin) !!!!!!!!!!")
                        logger.debug(f"Existing text: {outputs[high_sim_idx]}")
                        logger.debug(f"New text: {output_text}")
                        logger.debug(f"Similarity: {similarity}")
                        logger.debug("!!!!!!!!!! Found similar embedding (End) !!!!!!!!!!")
                        continue
                else:
                    output_text = self.base_model(txt_or_msg, role=create_role("policy", query_idx, from_phase), temperature=temperature, \
                        max_length=self.max_length, max_new_tokens=self.max_new_tokens, top_p=self.top_p, \
                        new_sent_stop=False, enable_thinking=False, return_embedding=False).text.strip()
                # output_dict = parse_reasoning_and_label(output_text) # use when enable_thinking=True BUT I need to figure output how to stop correctly
                # output_text = output_dict['label'] if output_dict['label'] is not None else ''
                
                tokens = self.base_model.tokenizer(output_text, return_tensors="pt", add_special_tokens=False).input_ids
                if tokens.shape[1] > 1000:
                    logger.debug(f"!!!!!!!!!! Output is larger than 1000 tokens (Begin) !!!!!!!!!!")
                    logger.debug(f"Output (temperature: {temperature}): {output_text}")
                    logger.debug("\nWith system prompt: ")
                    logger.debug(self.task_prompt_spec)
                    logger.debug("\nWith user prompt: ")
                    logger.debug(txt_or_msg)
                    logger.debug("!!!!!!!!!! Output is larger than 1000 tokens (End) !!!!!!!!!!")
                    continue
                    
                # check whether the prefix is "Next step: "
                if output_text.startswith("Next step: "):
                    output_text = output_text[11:]

                # check whether the prefix is "Step #" where # is the number of existing steps
                if re.match("Step \d+:", output_text) is not None: 
                    matched_text = re.match("Step \d+:", output_text)[0]
                    output_text = output_text[len(matched_text):].strip()
                
                
                # check whether example or any of previous step(s) is in output_text
                if output_text not in existing_steps and query not in output_text:
                    break
                else:
                    logger.debug(f"!!!!!!!!!! Output is in existing steps (Begin) !!!!!!!!!!")
                    logger.debug(f"Output (temperature: {temperature}): {output_text}")
                    logger.debug("\nWith system prompt: ")
                    logger.debug(self.task_prompt_spec)
                    logger.debug("\nWith user prompt: ")
                    logger.debug(txt_or_msg)
                    logger.debug("!!!!!!!!!! Output is in existing steps (End) !!!!!!!!!!")
                    n_retry_repeat += 1
                    if n_retry_repeat > 2:
                        output_text = "ALWAY REPEAT. TERMINATE"
                        break
            outputs.append(output_text)
            if self.check_action_sim:
                embeddings.append(embedding)

        return outputs