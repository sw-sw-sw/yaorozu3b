2024-09-01
⌘ SHIFT V = Preview

### マルチプロセッシングを継承することで決定 ok



### 本質的な構造で効率化できるところはないか？



### visual_systemは今後のために
　while ループを外部化する

### add_agent、remove_agentsの実装
　そのためのshare_memoryの管理方法を検討する

### エージェントがランダムに増える、減る、を実装する。

### 衝突検知を入れる

### 捕食処理を入れる

### 増殖処理を入れる





### 追加、ブラッシュアップ事項


### VisualSystemを独自にshare_memoryから読み込んで描画するようにする


### TensorFlowSimulationの構造をself.positionsを持たない方法に変える？

今後、エージェントの増減があった場合、active_agentsの数は変動するので、シンクが失敗するとエラーが起きる可能性があります。
現在、TensorFlowSimulationでは内部にself.positionsを保有していますが、
毎回共有メモリーから取得し、self.positionsを持たない方がエラーが出ないのではないでしょうか？
ただ内部にself.positionsを持たないことによって、効率が悪くなるようなことはありますか？
毎回メモリーを確保することでメモリー不足になるようなことはありますか？

↓

パフォーマンスの測定をする？