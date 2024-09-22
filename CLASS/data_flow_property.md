```mermaid
graph TB
    subgraph Ecosystem

        E_AD[AgentsData<br><br>positions<br>velocities<br>species<br>agent_ids<br>current_agent_count]
        E_AD_UP[AD.update]
        E_ARA[add_random_agent]
        E_RRA[remove_random_agents]

        E_AD_UP --> E_AD
        E_ARA --> E_AD
        E_RRA --> E_AD
    end

    subgraph TensorFlowSimulation

        TF_UP_P[update_property<br><br>tf_positions<br>tf_species<br>tf_current_agent_count]
        TF_CF[calculate_forces<br><br>tf_forces]
        
        TF_UP_P --> TF_CF


    end

    subgraph Box2DSimulation

        B2D_STEP[step<br><br>world]
        B2D_PEQ[process_ecosystem_queue]
        B2D_HAA[_handle_agent_added<br><br>bodies]
        B2D_HAR[_handle_agent_removed<br><br>bodies]
        B2D_STF[send_data_to_tf<br><br>positions<br>species<br>current_agent_count]
        B2D_STE[send_data_to_eco<br><br>positions<br>agent_ids<br>current_agent_count]
        B2D_UF[update_forces<br><br>bodies]
        B2D_UP_P[update_positions<br><br>positions]

        B2D_HAR --> B2D_STEP
        B2D_HAA --> B2D_STEP
        B2D_PEQ --> B2D_HAA 
        B2D_PEQ --> B2D_HAR
        B2D_UF --> B2D_STEP
        B2D_STEP -->  B2D_UP_P
        B2D_UP_P --> B2D_STF
        B2D_UP_P --> B2D_STE
        
    end

    subgraph VisualSystem
 

        VS_PQ[process_queue<br>add/remove]
        VS_HAA[_handle_agent_added<br><br>creatures]
        VS_HAR[_handle_agent_removed<br><br>creatures]
        VS_UB[update_buffer<br>main-loop<br><br>positions<br>agent_ids<br>current_agent_count]
        VS_UC[update_creatures]
        VS_CRE[Creatures<br><br>position<br>species]

        VS_HAA --> VS_CRE
        VS_HAR --> VS_CRE
        VS_UB --> VS_UC
        VS_UC --> VS_CRE

        VS_PQ --> VS_HAA
        VS_PQ --> VS_HAR
    end

    %% Inter-class data flows
    E_AD_UP -.->|action, agent_id, species, position, velocity| B2D_PEQ
    B2D_STF -->|positions, species, current_agent_count| TF_UP_P
    B2D_STE -->|positions, agent_ids, current_agent_count| E_AD_UP
    TF_CF -->|forces, current_agent_count| B2D_UF
    E_AD -->|action, agent_id, species, position| VS_PQ
    E_AD -->|positions, agent_ids, current_agent_count| VS_UB
    ```