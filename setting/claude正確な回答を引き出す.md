Claudeから正確な回答を引き出す方法


## 1. 重要な前提条件や制約のリマインダー:



具体例:
```
# 重要な前提条件:
- マルチプロセッシングを使用
- 共有メモリをクラスに含めない
- MAX_AGENTS_NUM = 3000
- Box2D, TensorFlow, Pygameを使用

# 最近の変更点:
- sparse_agent_arrayモジュールの導入
- ParameterControlUIの追加

質問: [ここに質問や要求を記述]
```

## 2. 段階的なタスク分割:



具体例:
```
タスク: Box2DSimulationクラスの更新

ステップ1: 初期化メソッドの更新
ステップ2: create_bodiesメソッドの修正
ステップ3: stepメソッドの調整
ステップ4: 新しいメソッドの追加（必要な場合）

現在のステップ: 1
質問: 初期化メソッドをsparse_agent_arrayに対応させるには、どのような変更が必要ですか？
```

## 3. 定期的なサマリー:



具体例:
```
# これまでの主な変更点:
1. TensorFlowSimulationクラスにpredator_prey_forcesメソッドを追加
2. Box2DSimulationクラスでsparse_agent_arrayを使用するように更新
3. VisualSystemクラスに新しいレンダリングオプションを追加

# 次のステップ:
- EcosystemクラスのSAA対応
- パフォーマンス最適化

質問: EcosystemクラスをSAAに対応させるために、どのメソッドを優先的に更新すべきですか？
```

## 4. 明示的な文脈の提供:



具体例:
```
# 前回のスレッドからの重要な文脈:
- sparse_agent_array (SAA)モジュールを導入
- Box2DSimulationクラスをSAA対応に更新
- TensorFlowSimulationクラスに種別ごとの相互作用を追加

# 現在の目標:
EcosystemクラスをSAAに対応させる

質問: Ecosystemクラスのinitialize_agentsメソッドをSAAに対応させるには、どのような変更が必要ですか？
```

## 5. 重要なコードブロックの保護:



具体例:
```python
<DO_NOT_MODIFY>
def calculate_forces(self, positions, species):
    species_forces = self._species_forces(positions, species)
    environment_forces = self._environment_forces(positions)
    return species_forces + environment_forces
</DO_NOT_MODIFY>

# 新しいメソッドを以下に追加してください
def new_method(self):
    # ここに新しいコードを記述
    pass
```

これらの方法を適切に組み合わせることで、より一貫性のある高品質な対話と開発支援が可能になると考えられます。どの方法が最も効果的か、また他に有効な方法があるかについて、フィードバックをいただければ幸いです。