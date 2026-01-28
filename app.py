import gradio as gr
import torch
import numpy as np
from PIL import Image
import os

# Import local modules
from dqn import DQN

# Configuration
MODEL_PATH = "runs/flappybird1.pt"
GRAPH_PATH = "runs/flappybird1.png"
LOG_PATH = "runs/flappybird1.log"

# Model parameters (from hyperparameters.yml)
STATE_DIM = 12  # Flappy Bird state features
ACTION_DIM = 2  # Do nothing or Flap
HIDDEN_DIM = 512
ENABLE_DUELING = True

def load_model():
    """Load the pre-trained DQN model."""
    model = DQN(STATE_DIM, ACTION_DIM, HIDDEN_DIM, ENABLE_DUELING)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        return model
    return None

def get_action_name(action_idx):
    """Convert action index to human-readable name."""
    return "Flap!" if action_idx == 1 else "Do Nothing"

def predict_action(
    player_y, player_vel, player_rot,
    last_pipe_x, last_top_y, last_bottom_y,
    next_pipe_x, next_top_y, next_bottom_y,
    next_next_pipe_x, next_next_top_y, next_next_bottom_y
):
    """Predict the best action given a game state."""
    model = load_model()
    if model is None:
        return "Model not loaded", "N/A", "Model file not found"

    # Construct state vector (normalized values expected)
    state = torch.tensor([
        last_pipe_x, last_top_y, last_bottom_y,
        next_pipe_x, next_top_y, next_bottom_y,
        next_next_pipe_x, next_next_top_y, next_next_bottom_y,
        player_y, player_vel, player_rot
    ], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state).squeeze()
        action = q_values.argmax().item()
        confidence = torch.softmax(q_values, dim=0)[action].item() * 100

    return (
        get_action_name(action),
        f"{confidence:.1f}%",
        f"Q-values: [Do Nothing: {q_values[0]:.3f}, Flap: {q_values[1]:.3f}]"
    )

def get_training_graph():
    """Return the training progress graph."""
    if os.path.exists(GRAPH_PATH):
        return Image.open(GRAPH_PATH)
    return None

def get_training_logs():
    """Return the training logs."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            return f.read()
    return "No training logs available."

def get_model_info():
    """Return model architecture info."""
    model = load_model()
    if model is None:
        return "Model not loaded"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = f"""
## Model Architecture: Dueling DQN

**State Features (12 inputs):**
- Last pipe: horizontal pos, top Y, bottom Y
- Next pipe: horizontal pos, top Y, bottom Y
- Next-next pipe: horizontal pos, top Y, bottom Y
- Player: vertical pos, velocity, rotation

**Actions (2 outputs):**
- 0: Do Nothing
- 1: Flap

**Network:**
- Hidden layer: {HIDDEN_DIM} nodes
- Dueling DQN: Enabled (separates value and advantage streams)
- Double DQN: Enabled (reduces Q-value overestimation)

**Parameters:**
- Total: {total_params:,}
- Trainable: {trainable_params:,}
"""
    return info

# Create Gradio interface
with gr.Blocks(title="Flappy Bird DQN Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Flappy Bird DQN Agent

    This agent was trained using **Deep Reinforcement Learning** with **Double DQN** and **Dueling DQN** architectures.

    The agent learns to play Flappy Bird by observing pipe positions and player state, then deciding whether to flap or not.
    """)

    with gr.Tab("Training Results"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Training Progress")
                graph_output = gr.Image(value=get_training_graph(), label="Rewards & Epsilon Decay")
            with gr.Column():
                gr.Markdown("### Training Logs")
                logs_output = gr.Textbox(value=get_training_logs(), label="Logs", lines=15, max_lines=20)

    with gr.Tab("Model Info"):
        gr.Markdown(get_model_info())

    with gr.Tab("Try the Agent"):
        gr.Markdown("### Input a game state to see what action the agent would take")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Player State**")
                player_y = gr.Slider(-1, 1, value=0, label="Player Y Position")
                player_vel = gr.Slider(-1, 1, value=0, label="Player Velocity")
                player_rot = gr.Slider(-1, 1, value=0, label="Player Rotation")

            with gr.Column():
                gr.Markdown("**Pipe Positions (normalized)**")
                next_pipe_x = gr.Slider(0, 1, value=0.5, label="Next Pipe X")
                next_top_y = gr.Slider(-1, 1, value=0.3, label="Next Pipe Top Y")
                next_bottom_y = gr.Slider(-1, 1, value=-0.3, label="Next Pipe Bottom Y")

        with gr.Row():
            with gr.Column():
                last_pipe_x = gr.Slider(-1, 1, value=-0.2, label="Last Pipe X")
                last_top_y = gr.Slider(-1, 1, value=0.3, label="Last Pipe Top Y")
                last_bottom_y = gr.Slider(-1, 1, value=-0.3, label="Last Pipe Bottom Y")
            with gr.Column():
                next_next_pipe_x = gr.Slider(0, 2, value=1, label="Next-Next Pipe X")
                next_next_top_y = gr.Slider(-1, 1, value=0.2, label="Next-Next Pipe Top Y")
                next_next_bottom_y = gr.Slider(-1, 1, value=-0.4, label="Next-Next Pipe Bottom Y")

        predict_btn = gr.Button("Predict Action", variant="primary")

        with gr.Row():
            action_output = gr.Textbox(label="Recommended Action")
            confidence_output = gr.Textbox(label="Confidence")
            qvalues_output = gr.Textbox(label="Q-Values")

        predict_btn.click(
            predict_action,
            inputs=[
                player_y, player_vel, player_rot,
                last_pipe_x, last_top_y, last_bottom_y,
                next_pipe_x, next_top_y, next_bottom_y,
                next_next_pipe_x, next_next_top_y, next_next_bottom_y
            ],
            outputs=[action_output, confidence_output, qvalues_output]
        )

if __name__ == "__main__":
    demo.launch()
