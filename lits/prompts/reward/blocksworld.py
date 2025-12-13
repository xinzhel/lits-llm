task_prompt_spec_blocksworld_rag = {
        "icl": """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do
Pick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.

[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.
My goal is to have that the orange block is on top of the red block.

My plan is as follows:

[PLAN]\nunstack the yellow block from on top of the orange block\nput down the yellow block\npick up the orange block\nstack the orange block on top of the red block
[PLAN END]

[STATEMENT]\nAs initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block, the red block is on the table and the yellow block is on the table.
My goal is to have that the blue block is on top of the red block and the yellow block is on top of the orange block.

My plan is as follows:

[PLAN]\npick up the yellow block\nstack the yellow block on top of the orange block
[PLAN END]

[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, the blue block is on top of the yellow block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the blue block is on top of the orange block and the yellow block is on top of the red block.

My plan is as follows:

[PLAN]\nunstack the blue block from on top of the yellow block\nstack the blue block on top of the orange block\npick up the yellow block\nstack the yellow block on top of the red block
[PLAN END]

[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the yellow block is on top of the orange block, the red block is on the table, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the blue block and the yellow block is on top of the red block.

My plan is as follows:

[PLAN]\nunstack the yellow block from on top of the orange block\nstack the yellow block on top of the red block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]

[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>

My plan is as follows:

[PLAN]\n<action>""",
    "evaluator": """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Pick up a block
Unstack a block from on top of another block
Put down a block\nStack a block on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.

Please evaluate whether the given action is a good one under certain conditions.

[STATEMENT]
As initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.
My goal is to have that the orange block is on top of the red block.
[ACTION]
unstack the red block from on top of the blue block
[EVALUATION]
bad

[STATEMENT]
As initial conditions I have that, the orange block is in the hand, the yellow block is clear, the hand is holding the orange block, the blue block is on top of the red block, the yellow block is on top of the blue block, and the red block is on the table.
My goal is to have have that the yellow block is on top of the orange block.
[ACTION]
put down the orange block
[EVALUATION]
good

[STATEMENT]
As initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block, the red block is on the table and the yellow block is on the table.
My goal is to have that the blue block is on top of the red block and the yellow block is on top of the orange block.
[ACTION]
pick up the yellow block
[EVALUATION]
good

[STATEMENT]
As initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block, the red block is on the table and the yellow block is on the table.
My goal is to have that the blue block is on top of the red block and the yellow block is on top of the orange block.
[ACTION]
pick up the yellow block
[EVALUATION]
good

[STATEMENT]
As initial conditions I have that, the blue block is clear, the orange block is in the hand, the red block is clear, the hand is holding the orange block, the red block is on top of the yellow block, the blue block is on the table, and the yellow block is on the table.
My goal is to have have that the red block is on top of the yellow block and the orange block is on top of the blue block.
[ACTION]
stack the orange block on top of the red block
[EVALUATION]
bad

[STATEMENT]
As initial conditions I have that, <init_state>
My goal is to <goals>
[ACTION]
<action>
[EVALUATION]
"""
}

task_prompt_spec_blocksworld = """I am operating in a BlocksWorld environment with colored blocks and a robot hand.
I must judge whether a single proposed action I perform is **good**, **bad**, or **unknown** under the initial conditions and the goal.

---

## **ACTIONS I CAN DO**

* pick up a block
* unstack a block from on top of another block
* put down a block
* stack a block on top of another block

---

## **RESTRICTIONS ON MY ACTIONS**

I must obey all of the following rules:

1. I can only pick up or unstack one block at a time.
2. I can only pick up or unstack a block if my hand is empty.
3. I can only pick up a block if it is on the table and is clear.
4. A block is clear if no block is on top of it and it is not being held.
5. I can only unstack a block from another block if the block was actually on top of that block.
6. I can only unstack a block if it is clear.
7. After I pick up or unstack a block, I am holding it.
8. I can only put down a block I am holding.
9. I can only stack a block on another block if I am holding the block being stacked.
10. I can only stack onto a block if that block is clear.
11. After I put down or stack a block, my hand becomes empty.

---

## **EVALUATION PRINCIPLES**

### **1. ACTION LEGALITY**

If the action violates any restriction, it is **bad**.

### **2. GOAL-DIRECTEDNESS**

If the action is legal, I evaluate whether it lies on a **reasonable shortest path** toward the goal.

* A legal action is **good** if it efficiently advances toward the goal or establishes a necessary precondition.
* A legal action is **bad** if it creates unnecessary extra steps, destroys correct structure, or ignores a direct solution path.

### **3. UNCERTAINTY**

If I cannot confidently evaluate goal-directedness, I return **unknown**.

---

## **OUTPUT FORMAT**

```
[REASONING]
<brief specific reasoning>

[EVALUATION]
good | bad | unknown
```

---

# **EXAMPLES**

---

### **Example 1**

[STATEMENT]
As initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.
My goal is to have that the orange block is on top of the red block.

[ACTION]
unstack the red block from on top of the blue block

[REASONING]
The action is legal because the hand is empty, red is clear, and red is on blue.
However, this action is not helpful: the goal requires placing **orange on red**, and I can achieve this directly by first removing yellow from orange, then picking up orange and stacking it onto red.
Unstacking red is unnecessary and breaks the existing blue–red structure, which is not an obstacle for achieving the goal.
This introduces extra steps and moves a block that does not need to be moved.

[EVALUATION]
bad

---

### **Example 2**

[STATEMENT]
As initial conditions I have that, the orange block is in the hand, the yellow block is clear, the hand is holding the orange block, the blue block is on top of the red block, the yellow block is on top of the blue block, and the red block is on the table.
My goal is to have that the yellow block is on top of the orange block.

[ACTION]
put down the orange block

[REASONING]
The action is legal because I am holding orange.
To reach the goal, I must place yellow on orange. But I cannot unstack yellow while holding orange, so the first necessary step is to empty my hand.
Putting orange down enables me to free yellow afterward, which is required before stacking it onto orange.
Thus this move is part of a shortest valid plan.

[EVALUATION]
good

---

### **Example 3**

[STATEMENT]
As initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block, the red block is on the table and the yellow block is on the table.
My goal is to have that the blue block is on top of the red block and the yellow block is on top of the orange block.

[ACTION]
pick up the yellow block

[REASONING]
The action is legal because the hand is empty, yellow is clear, and yellow is on the table.
The goal already has **blue on red**, so that part needs no modification.
The remaining goal requirement is **yellow on orange**, and picking up yellow is exactly the necessary first step before stacking it onto orange.
This action directly advances the unfinished part of the goal.

[EVALUATION]
good

---

### **Example 4**

[STATEMENT]
As initial conditions I have that, the blue block is clear, the orange block is in the hand, the red block is clear, the hand is holding the orange block, the red block is on top of the yellow block, the blue block is on the table, and the yellow block is on the table.
My goal is to have that the red block is on top of the yellow block and the orange block is on top of the blue block.

[ACTION]
stack the orange block on top of the red block

[REASONING]
The action is legal because I am holding orange and red is clear.
But this action moves orange farther from blue, even though I can **achieve the goal immediately** by stacking orange directly onto the clear blue block.
Stacking orange on red forces me to later unstack orange again before achieving the goal, adding unnecessary steps and working against a direct solution.

[EVALUATION]
bad
---"""

usr_prompt_spec_blocksworld = """[STATEMENT]
As initial conditions I have that, <init_state>
My goal is to <goals>
[ACTION]
<action>
[EVALUATION]"""