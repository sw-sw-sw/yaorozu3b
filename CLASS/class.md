
### Alifeシュミレーターの構成です。
・マルチプロセッシングと共有メモリーを使用しています。
・それぞれのメソッド、それぞれのクラスはマルチプロセッシングで動いています。
・そのため、mainファイルの各メソッド内(tf_loop、box2d_loop、visual_system_loop、ecosystem_liip)で、各クラスをインスタンス化しています。
・SharedMemoryManagerクラスはmainファイル内でインスタンス化されています。
・それぞれのクラスはSharedMemoryManagerでデーターを共有しています。
・SharedMemoryManagerのインスタンスは、各プロセスに渡され、さらに各クラスのイニシャライザーに渡されています。
・それによって、大量のシミュレーションを効率よく処理するシステムになっています。
・各クラスはマルチプロセスによって並列動作しているため、クラス間のやりとりはqueueによって行われています。

```
[Main Process]
  |
  |--> Creates SharedMemoryManager
  |--> Creates Queues
  |
  |--> Spawns Process: [Ecosystem Process]
  |     |- Ecosystem
  |     |- Uses SharedMemoryManager
  |     |- Uses Queues: ecosystem_to_box2d, ecosystem_to_visual, box2d_to_eco, visual_to_eco
  |
  |--> Spawns Process: [Box2D Process]
  |     |- Box2DSimulation
  |     |- Uses SharedMemoryManager
  |     |- Uses Queues: ecosystem_to_box2d, box2d_to_eco
  |
  |--> Spawns Process: [TensorFlow Process]
  |     |- TensorFlowSimulation
  |     |- Uses SharedMemoryManager
  |
  |--> Spawns Process: [Visual System Process]
       |- VisualSystem
       |- Uses SharedMemoryManager
       |- Uses Queues: ecosystem_to_visual, visual_to_eco

[SharedMemoryManager]
  |- Manages shared memory for positions, velocities, forces, etc.

[Queues]
  |- ecosystem_to_box2d
  |- box2d_to_eco
  |- ecosystem_to_visual
  |- visual_to_eco

```

  ```mermaid
classDiagram
    class Main {
        -shared_memory_manager: SharedMemoryManager
        -queue_eco_to_box2d: Queue
        -queue_eco_to_visual: Queue
        +run_ecosystem(shared_memory_manager, queue_eco_to_box2d, queue_eco_to_visual)
        +run_box2d(shared_memory_manager, queue_eco_to_box2d)
        +run_visual_system(shared_memory_manager, queue_eco_to_visual)
    }

    class SharedMemoryManager {
        -positions: np.ndarray
        -velocities: np.ndarray
        -forces: np.ndarray
        +get_positions()
        +update_positions(new_positions)
        +get_velocities()
        +update_velocities(new_velocities)
        +get_forces()
        +update_forces(new_forces)
    }

    class Ecosystem {
        -shared_memory: SharedMemoryManager
        -queue_to_box2d: Queue
        -queue_to_visual: Queue
        +run()
    }

    class Box2DSimulation {
        -shared_memory: SharedMemoryManager
        -queue_from_eco: Queue
        +run()
    }

    class TensorflowSystem {
        -shared_memory: SharedMemoryManager
        +run()
    }

    class VisualSystem {
        -shared_memory: SharedMemoryManager
        -queue_from_eco: Queue
        +run()
    }

    Main --> SharedMemoryManager : creates
    Main --> Ecosystem : spawns process
    Main --> Box2DSimulation : spawns process
    Main --> VisualSystem : spawns process
    Main --> TensorflowSystem : spawn process
    Ecosystem --> SharedMemoryManager : uses
    Box2DSimulation --> SharedMemoryManager : uses
    VisualSystem --> SharedMemoryManager : uses
    TensorflowSystem --> SharedMemoryManager : uses
    Ecosystem ..> Box2DSimulation : communicates via queuex
    Ecosystem ..> VisualSystem : communicates via queue
```

.
.
.
以下は、上記のmainファイルとその中のマルチプロセッシングに関わる各プロセスを省略しています。
実際の機能に関わるクラスだけを取り出したクラス図です。
各クラスで追次処理が必要な場合は、multiprocessing.Queueによって、連携を取ります。
以下の点線は、Queueの流れです。
・
・
・
```mermaid
classDiagram

    class Ecosystem {
        -sharedMemoryManager: SharedMemoryManager
        -box2DSimulation: Box2DSimulation
        -tensorFlowSimulation: TensorFlowSimulation
        +__init__(max_agents: int, world_width: float, world_height: float)
        +step()
        +updateAllAgents()
        +manageBirth()
        +manageDeath()
        +managePredation()
        +updatePopulation()
        +getPopulationStatistics() : PopulationStatistics
        -createNewAgent(species: String, position: Vector2D) : int
        -removeAgent(index: int)
        -initializeSimulations()
    }

    class SharedMemoryManager {
        -max_agents: int
        -dimensions: int
        -positions: mp.Array
        -velocities: mp.Array
        -forces: mp.Array
        -agent_ids: mp.Array
        -agent_species: mp.Array
        -active_mask: mp.Array

        -np_positions: np.ndarray
        -np_velocities: np.ndarray
        -np_forces: np.ndarray
        -np_agent_ids: np.ndarray
        -np_agent_species: np.ndarray
        -np_active_mask: np.ndarray
        -current_agent_count: mp.Value
        -lock: mp.Lock

        +__init__(max_agents: int, dimensions: int)
        +add_agent(position: np.ndarray, velocity: np.ndarray, species: int) : int
        +remove_agent(index: int) : bool
        +get_active_agents() : tuple

        +get_positions_for_box2d() : List[Tuple[float, float]]
        +get_data_for_tensorflow() : Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        +get_data_for_visual_system() : Tuple[np.ndarray, np.ndarray, np.ndarray]

        +update_positions(new_positions: np.ndarray)
        +update_forces(new_forces: np.ndarray)
    }






    class Box2DSimulation {
        -shared_memory: SharedMemoryManager
        -world: b2World
        -bodies: List[b2Body]

        +__init__(shared_memory_manager: SharedMemoryManager)
        +step()
        +applyForces(forces: List[Tuple[float, float]])
        -update_shared_memory(positions: List)
        +initializeAgent(index: int, agent: AgentData)
        +removeAgent(index: int)
        -createBody(index: int, agent: AgentData): b2Body
        -updateBodyProperties(index: int, agent: AgentData)
        +swapBodies(index1: int, index2: int)
    }

    class TensorFlowSimulation {
        -world_size: tf.Tensor
        -num_agents: int
        -positions: tf.Variable
        -max_force: tf.Tensor
        -separation_distance: tf.Variable
        -alignment_distance: tf.Tensor
        -cohesion_distance: tf.Tensor
        -separation_weight: tf.Tensor
        -alignment_weight: tf.Tensor
        -cohesion_weight: tf.Tensor
        +__init__(num_agents: int, world_width: float, world_height: float)
        +update_positions(new_positions: tf.Tensor)
        +set_initial_positions(initial_positions: tf.Tensor)
        +calculate_forces() : tf.Tensor
        -precompute_distances() : tf.Tensor
        -separation(distances: tf.Tensor) : tf.Tensor
        -alignment(distances: tf.Tensor) : tf.Tensor
        -cohesion(distances: tf.Tensor) : tf.Tensor
        -limit_magnitude(vectors: tf.Tensor, max_magnitude: tf.Tensor) : tf.Tensor
    }

    class VisualSystem {
        -active_mask: np.Array
        +draw(positions: Vector2D[] )
        +create_creature(positions: np.arrays, agent_id: int, agent_species_id : int)
        -remove_creature()

        +create_death_effect(positions: np.arrray)
        +render_creatures(positions: np.arrays)
    }

    class Creature {
    +update()
    +draw()
    +creature[]
    
    }

    class DeathEffect {
    +update()
    +draw()
    +death_effect[]
    }

    Ecosystem --> SharedMemoryManager: uses
    Ecosystem ..> VisualSystem: [queue] \n\n add_agent(\n position, \n agent_id, \n agent_species\n)\n -> agent_radius \n\n\n remove_agent(agent_id)\n -> step complete
    Ecosystem ..> Box2DSimulation: [queue] \n create_agent ( \n    positions, \n    velocity, \n    agent species, \n    agent_id, \n    agent_radius \n)-> step complete \n\n\n remove agent(agent_id) \n -> step complete
    DeathEffect ..> Ecosystem: [queue] \n death_effect_finished(agent_id) -> True

    SharedMemoryManager --> TensorFlowSimulation : positions tf.Tensor\nagent_ids tf.Tensor\nagent_species tf.Tensor
    SharedMemoryManager <-- TensorFlowSimulation : forces tf.Tensor
    SharedMemoryManager --> Box2DSimulation : forces \nList[Tuple[float, float]]
    SharedMemoryManager <-- Box2DSimulation : positions \nList[Tuple[float, float]]
    SharedMemoryManager --> VisualSystem: positions \n agent_id \n agent_species \n active_mask \n\n for render


    VisualSystem --> Creature: positions \nagent_id \nagent_species
    VisualSystem --> DeathEffect: positions

```

