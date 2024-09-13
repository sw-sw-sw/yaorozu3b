from agents_data import AgentsData
import numpy as np
from queue import Queue

def test_agents_data():
    max_agents = 10
    queue_dict = {
        'eco_to_box2d': Queue(),
        'eco_to_visual': Queue(),
        'eco_to_box2d_init': Queue(),
        'eco_to_visual_init': Queue(),
        'eco_to_tf_init': Queue(),
        'box2d_to_eco': Queue(),
        'eco_to_visual_render': Queue()
    }
    
    agents_data = AgentsData(max_agents, queue_dict)

    print("1. Adding agents:")
    for i in range(5):
        agent_id = agents_data.add_agent(species=i % 3, position=(i * 10, i * 10))
        print(f"Added agent {agent_id}: species {i % 3}, position ({i * 10}, {i * 10})")

    print("\nCurrent state:")
    print(f"Agent IDs: {agents_data.agent_ids[:agents_data.current_agent_count]}")
    print(f"Positions:\n{agents_data.positions[:agents_data.current_agent_count]}")
    print(f"Species: {agents_data.species[:agents_data.current_agent_count]}")

    print("\n2. Removing agents:")
    agents_data.remove_agent(2)
    print("Removed agent 2")
    agents_data.remove_agent(4)
    print("Removed agent 4")

    print("\nCurrent state after removals:")
    print(f"Agent IDs: {agents_data.agent_ids[:agents_data.current_agent_count]}")
    print(f"Positions:\n{agents_data.positions[:agents_data.current_agent_count]}")
    print(f"Species: {agents_data.species[:agents_data.current_agent_count]}")

    print("\n3. Adding more agents:")
    for i in range(3):
        agent_id = agents_data.add_agent(species=i % 2, position=(i * 20, i * 20))
        print(f"Added agent {agent_id}: species {i % 2}, position ({i * 20}, {i * 20})")

    print("\nCurrent state after additions:")
    print(f"Agent IDs: {agents_data.agent_ids[:agents_data.current_agent_count]}")
    print(f"Positions:\n{agents_data.positions[:agents_data.current_agent_count]}")
    print(f"Species: {agents_data.species[:agents_data.current_agent_count]}")

    print("\n4. Attempting to add agent when at max capacity:")
    for _ in range(5):
        agent_id = agents_data.add_agent(species=0, position=(0, 0))
        if agent_id is None:
            print("Failed to add agent: maximum capacity reached")
        else:
            print(f"Added agent {agent_id}")

    print("\n5. Attempting to remove non-existent agent:")
    agents_data.remove_agent(100)

    print("\nFinal state:")
    print(f"Agent IDs: {agents_data.agent_ids[:agents_data.current_agent_count]}")
    print(f"Current agent count: {agents_data.current_agent_count}")
    print(f"Positions:\n{agents_data.positions[:agents_data.current_agent_count]}")
    print(f"Species: {agents_data.species[:agents_data.current_agent_count]}")

    print("\n6. Testing update_from_box2d and send_data_to_visual:")
    box2d_data = {
        'positions': np.array([(i, i) for i in range(agents_data.current_agent_count)]),
        'current_agent_count': agents_data.current_agent_count
    }
    queue_dict['box2d_to_eco'].put(box2d_data)
    updated = agents_data.update_from_box2d()
    print(f"Data updated: {updated}")

    print("Updated positions:")
    print(agents_data.positions[:agents_data.current_agent_count])
    
    visual_data = agents_data.send_data_to_visual()
    print("Data sent to visual system:", visual_data)

    updated = agents_data.update_from_box2d()
    print(f"\nData updated when no new data: {updated}")
    visual_data = agents_data.send_data_to_visual()
    print("Data sent to visual system when no update:", visual_data)

if __name__ == "__main__":
    test_agents_data()