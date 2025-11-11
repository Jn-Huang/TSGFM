"""
Script to load and inspect the Amazon Ratings dataset (smallest e-commerce dataset).
This script will download cache if needed and inspect node text information.
"""

import os
import sys
import torch
import pickle
from utils import SentenceEncoder, load_yaml
from task_constructor import UnifiedTaskConstructor

def inspect_amazonratings():
    print("=" * 80)
    print("Loading Amazon Ratings Dataset (Smallest E-commerce Dataset)")
    print("=" * 80)
    print(f"Dataset stats from README:")
    print(f"  - Nodes: 24,492")
    print(f"  - Edges: 186,100")
    print(f"  - Classes: 5 (rating categories)")
    print(f"  - Domain: E-commerce")
    print("=" * 80)

    # Initialize encoder
    print("\n[1/4] Initializing text encoder (minilm)...")
    encoder = SentenceEncoder("minilm", root="cache_data", batch_size=256)

    # Load configurations
    print("\n[2/4] Loading configuration files...")
    task_config = load_yaml("configs/task_config.yaml")
    data_config = load_yaml("configs/data_config.yaml")

    # Create task constructor
    print("\n[3/4] Creating task constructor for 'amazonratings'...")
    tasks = UnifiedTaskConstructor(
        task_names=['amazonratings'],
        encoder=encoder,
        task_config_lookup=task_config,
        data_config_lookup=data_config,
        root="cache_data_minilm",
        batch_size=32,
        sample_size=-1,
        node_centered=True
    )

    # Construct dataset (will download if needed)
    print("\n[4/4] Constructing dataset (downloading cache if not available)...")
    tasks.construct_exp()

    print("\n" + "=" * 80)
    print("Dataset loaded successfully! Inspecting data...")
    print("=" * 80)

    # Access the loaded dataset
    dataset_name = 'amazonratings'
    cache_path = f"cache_data_minilm/{dataset_name}"

    # Load the processed data
    processed_path = os.path.join(cache_path, "processed")

    if os.path.exists(os.path.join(processed_path, "geometric_data_processed.pt")):
        print(f"\n✓ Cache found at: {processed_path}")

        # Load the main data
        data_file = os.path.join(processed_path, "geometric_data_processed.pt")
        data = torch.load(data_file)

        print(f"\n[DATA STRUCTURE]")
        print(f"  Type: {type(data)}")
        if hasattr(data, 'keys'):
            print(f"  Keys: {data.keys()}")

        # Load texts
        texts_file = os.path.join(processed_path, "texts.pkl")
        if os.path.exists(texts_file):
            with open(texts_file, 'rb') as f:
                texts = pickle.load(f)

            print(f"\n[TEXT INFORMATION]")
            print(f"  Text structure type: {type(texts)}")
            print(f"  Number of text categories: {len(texts)}")

            # Inspect each text category
            text_categories = [
                "Node Features (Product Names)",
                "Edge Features (Relationships)",
                "NOI Node (Task Description)",
                "Class Nodes (Rating Categories)",
                "Prompt Edge Features"
            ]

            for i, (category_name, text_list) in enumerate(zip(text_categories, texts)):
                print(f"\n  [{i}] {category_name}")
                print(f"      - Type: {type(text_list)}")
                print(f"      - Count: {len(text_list)}")

                # Show first few examples
                if isinstance(text_list, list):
                    print(f"      - First 3 examples:")
                    for j, text in enumerate(text_list[:3]):
                        if isinstance(text, str):
                            preview = text[:100] + "..." if len(text) > 100 else text
                            print(f"        [{j}]: {preview}")
                        else:
                            print(f"        [{j}]: {text}")

                    if i == 0 and len(text_list) > 3:  # Show more for node features
                        print(f"      - Sample from middle and end:")
                        middle_idx = len(text_list) // 2
                        preview = text_list[middle_idx][:100] + "..." if len(text_list[middle_idx]) > 100 else text_list[middle_idx]
                        print(f"        [middle {middle_idx}]: {preview}")

                        last_idx = len(text_list) - 1
                        preview = text_list[last_idx][:100] + "..." if len(text_list[last_idx]) > 100 else text_list[last_idx]
                        print(f"        [last {last_idx}]: {preview}")

        # Get the actual OFA dataset object
        print(f"\n[GRAPH DATA]")
        ofa_dataset = tasks.data_lookup.get(dataset_name)
        if ofa_dataset:
            graph_data = ofa_dataset.get(0)
            print(f"  Graph data type: {type(graph_data)}")
            print(f"  Number of nodes: {graph_data.num_nodes}")
            print(f"  Number of edges: {graph_data.num_edges}")

            if hasattr(graph_data, 'y'):
                print(f"  Labels shape: {graph_data.y.shape}")
                print(f"  Unique labels: {torch.unique(graph_data.y).tolist()}")

            if hasattr(graph_data, 'node_text_feat'):
                print(f"  Node text features shape: {graph_data.node_text_feat.shape}")
                print(f"  Text embedding dimension: {graph_data.node_text_feat.shape[1]}")

            if hasattr(graph_data, 'class_node_text_feat'):
                print(f"  Class text features shape: {graph_data.class_node_text_feat.shape}")
                print(f"  Number of classes: {graph_data.class_node_text_feat.shape[0]}")

            # Check train/val/test splits
            if hasattr(graph_data, 'train_mask'):
                print(f"\n[SPLITS]")
                print(f"  Train nodes: {graph_data.train_mask.sum().item()}")
                print(f"  Val nodes: {graph_data.val_mask.sum().item()}")
                print(f"  Test nodes: {graph_data.test_mask.sum().item()}")
                total = graph_data.train_mask.sum() + graph_data.val_mask.sum() + graph_data.test_mask.sum()
                print(f"  Total split nodes: {total.item()}")
                print(f"  Train ratio: {graph_data.train_mask.sum().item() / graph_data.num_nodes:.2%}")
                print(f"  Val ratio: {graph_data.val_mask.sum().item() / graph_data.num_nodes:.2%}")
                print(f"  Test ratio: {graph_data.test_mask.sum().item() / graph_data.num_nodes:.2%}")

        print("\n" + "=" * 80)
        print("Inspection complete!")
        print("=" * 80)

    else:
        print(f"\n✗ Cache not found at: {processed_path}")
        print("  The dataset should have been created by the task constructor.")
        print("  Check if there were any errors during dataset construction.")

if __name__ == "__main__":
    try:
        inspect_amazonratings()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
