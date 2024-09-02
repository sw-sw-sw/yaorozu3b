2024-09-01
⌘ SHIFT V = Preview


# add_agent、remove_agentsの実装

### そのためのshare_memoryの管理方法  

データー構造の検討  
https://claude.ai/chat/0a7b0217-b3f0-4b0b-b7f3-6bda09948ed7  

`
positions  
agent_id  
active_mask`  
のセット  

イベントによる処理  
構造化配列  
スパーステンソルの利用  

### 共有メモリーの分離  
https://claude.ai/chat/79dff1e3-d66f-45c9-9c42-d53031f13143  
queueは別メモリーになっている。  

### TensorFlowSimulationの構造をself.positionsを持たせる。  

---

### エージェントがランダムに増える、減る、を実装する。
### 衝突検知を入れる
### 捕食処理を入れる
### 増殖処理を入れる
