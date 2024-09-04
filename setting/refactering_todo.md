2024-09-01
⌘ SHIFT V = Preview

# `add_agent`、`remove_agents`の実装
# データー構造の検討  

- claude  
https://claude.ai/chat/0a7b0217-b3f0-4b0b-b7f3-6bda09948ed7  

- 共有メモリーの分離  
queueは別メモリーになっている。  
https://claude.ai/chat/79dff1e3-d66f-45c9-9c42-d53031f13143  


---


# 2024-09-03 diary

- UIをブラッシュアップ  
- `TensorFlowSimulationの` `_predator_prey_forces`を修正動くようになった。  

- 共有メモリー構造のアイデア  
  

```
固定長3000のスパース行列を作る

agent_data={
        position
        Velocity
        Agent_id
        Species
        }   
の五種類

Count

————————

各クラスは内部コピーを持つ
定期的にアップデイトする

基本はキューを受けてアップデイトする
　最新状態のキューを一つだけ保持する

Tensorflowのforce=>box2dに送る、queueを受け取って、内部のfoeceをアップデイトする。

box2dのpositions 
=> tensorflowに送る。tensorflowは内部のpositionsをアップデイトする。
=> render_queueに送る。順次、描画する。
　creatureの点滅も追加する。
　将来、描画をprocessingに移す？

visual_systemはrender_queueをもらって自分の内部状態を更新する
　creatureの辞書のポジションを更新する
　　add/remove queueで処理する

ecoは、add/remove作業をqueueを使ってする

過不足があっても内部状態を元にアップデイトする

一旦コピーしてしまえば、メモリー競合は最小限になる
　
そうするとパラレルに動作できる
```
# forcesの整理
forecesは、それぞれの力の比率が問題
species forecesとenvironment forcesの割合を1対1にスケーリングする。
いらないforeceは何か？


---

# 残りタスク
- エージェントがランダムに増える、減る、を実装する。
- 衝突検知を入れる
- 捕食処理を入れる
- 増殖処理を入れる
