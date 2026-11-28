// データセット (ML-EDA の model_inputs) の入力と出す値の数を, 定義に入れる

import { parseRef, renameItem, specWires, type InputDef, type Spec, type UnitSpec } from './graph'

/** データセットの入力 (形はバッチの次元を例の大きさにしたもの. dataset_shape は決まらない次元が null の形) */
export interface DatasetInput {
  name: string
  shape: number[]
  dtype: string
  dataset_shape?: (number | null)[]
  /** 文の語の番号の入力の語彙の大きさ (語の番号の上限 + 1. 文でなければ無い) */
  vocabulary?: number | null
}

/** 表の生成の符号化した行の幅と内訳 (ML-Quest の table_encoding) */
export interface TableWidth {
  width: number
  /** 数の列の数 (1 ずつ) */
  numbers: number
  /** 種類の列の数 */
  categories: number
  /** 種類の列の値の種類の数の和 */
  values: number
  /** 学習に使う列の数 */
  columns: number
  /** 問題の設定の名前 */
  problem?: string | null
}

/** まとめるの符号化した行の幅と内訳 (ML-EDA の describe_groups. ML-Fathom の群の要約と同じ数え方) */
export interface GroupWidth {
  /** 符号化した行の幅 (入力 x の幅) */
  width: number
  /** 数の列の数 (1 ずつ) */
  numbers: number
  /** 種類の列の数 */
  categories: number
  /** 種類の列の値の種類の数の和 */
  values: number
  /** 群を作る列の数 */
  columns: number
  /** 問題の設定の名前 */
  problem?: string | null
}

/** 並べるの学習用の利用者と品物の数と決め方 (ML-Quest の rank_encoding) */
export interface RankSize {
  /** 学習用の利用者の数 (入力 user の大きさ) */
  users: number
  /** 学習用の品物の数 (入力 item の大きさ) */
  items: number
  /** 分割の名前から反応の行の数 (時刻の順に分けたもの) */
  rows: Record<string, number>
  /** プロジェクトの config の検証用の割合 (書いていなければ無い. 学習のスクリプトの既定) */
  valid?: number | null
  /** プロジェクトの config のテスト用の割合 (書いていなければ無い. 学習のスクリプトの既定) */
  test?: number | null
  /** 除いた行の知らせ */
  notes?: string[]
  /** 問題の設定の名前 */
  problem?: string | null
}

/** 異常の検知の入力と出力の幅 (学習に使う列の数) と決め方 */
export interface AnomalyWidth {
  /** 学習に使う列の数 (正解の列と行の id の列は数えない) */
  columns: number | null
  /** 異常の正解の列 (正解の無い表は無い) */
  target?: string | null
  /** 問題の設定の度合いの決め方 (autoencoder か distance. 問題の設定を渡していなければ無い) */
  score?: string | null
  /** 問題の設定の名前 */
  problem?: string | null
}

/** 効果を見積もるの入力 x の幅 (処置の列を除いた学習に使う列の数) と数えない列 */
export interface UpliftWidth {
  /** 学習に使う列の数 (処置の列, 結果の列と行の id の列は数えない) */
  columns: number | null
  /** 結果の列 (データセットの目的の列) */
  outcome?: string | null
  /** 問題の設定の処置の列 (問題の設定を渡していないか答えていなければ無い) */
  treatment?: string | null
  /** 問題の設定の行の id の列 (問題の設定を渡していないか答えていなければ無い) */
  id?: string | null
  /** 問題の設定の名前 */
  problem?: string | null
}

/** 値段を変える判断の入力 x の幅 (商品の特徴の列と値段の列の数) と列の名前 */
export interface PriceWidth {
  /** 特徴の列と値段の列の数 (売れた数, 原価, 商品の id と行の id の列は数えない) */
  columns: number | null
  /** 値段の列 (入力の最後の列. 分からなければ無い) */
  price?: string | null
  /** 売れた数の列 (データセットの目的の列) */
  units?: string | null
  /** 原価の列 (無いか分からなければ無い) */
  cost?: string | null
  /** 商品の id の列 (無いか分からなければ無い) */
  product?: string | null
  /** 行の id の列 (分からなければ無い) */
  id?: string | null
  /** 問題の設定の名前 */
  problem?: string | null
}

/** 時間をまたぐ量の判断の入力 x の幅 (学習に使う列の数), 数えない列と学ぶ分位点の水準 */
export interface DemandWidth {
  /** 学習に使う列の数 (日ごとの売れた数, 商品の id, 日付, 在庫の数と行の id の列は数えない) */
  columns: number | null
  /** 日ごとの売れた数の列 (分からなければ無い) */
  units?: string | null
  /** 商品の id の列 (分からなければ無い) */
  product?: string | null
  /** 日付の列 (分からなければ無い) */
  date?: string | null
  /** 在庫の数の列 (無いか分からなければ無い) */
  stock?: string | null
  /** 行の id の列 (分からなければ無い) */
  id?: string | null
  /** 学ぶ分位点の水準 (出力 quantiles の列の順. 問題の設定を渡していないか読めなければ無い) */
  levels?: number[] | null
  /** 水準をどこから読んだか (config は学習の設定, problem は問題の設定) */
  levels_from?: string | null
  /** 水準の決め方の知らせ (学習の設定と問題の設定の水準の数の食い違いなど) */
  notes?: string[]
  /** 問題の設定の名前 */
  problem?: string | null
}

/** バックエンドが ML-EDA の model_inputs から返す値 */
export interface DatasetInfo {
  path: string
  inputs: DatasetInput[]
  task?: string | null
  target?: string | null
  classes?: unknown[] | null
  /** ファイルの組の正解の種類 (mask, point_cloud など. 列の正解は null) */
  target_kind?: string | null
  outputs?: number | null
  /** 表の生成の問題の設定で, 入力と出力の幅にした符号化した行の幅 (他は無い) */
  table?: TableWidth | null
  /** 並べるの問題の設定で, 入力 user と item の大きさにした学習用の利用者と品物の数 (他は無い) */
  rank?: RankSize | null
  /** まとめるの問題の設定で, 入力 x の幅にした符号化した行の幅 (他は無い) */
  group?: GroupWidth | null
  /** 異常の検知のデータセットで, 入力と出力の幅にした学習に使う列の数と度合いの決め方 (他は無い) */
  anomaly?: AnomalyWidth | null
  /** 効果を見積もるのデータセットで, 入力 x の幅にした学習に使う列の数と数えない列 (他は無い) */
  uplift?: UpliftWidth | null
  /** 値段を変える判断のデータセットで, 入力 x の幅にした特徴の列と値段の列の数と列の名前 (他は無い) */
  price?: PriceWidth | null
  /** 需要の記録のデータセットで, 入力 x の幅にした学習に使う列の数, 数えない列と学ぶ水準 (他は無い) */
  demand?: DemandWidth | null
  /**
   * 入力の名前から, 骨組みがデータセットの入力の最後の次元に足す数 (ML-Quest の約束の入力の offset.
   * 効果の見積もりの S-learner の処置の列の 1. 足さない入力は無い)
   */
  offsets?: Record<string, number>
}

/** 入れた結果 (定義と, 知らせる文) */
export interface DatasetApplied {
  spec: Spec
  notes: string[]
}

/** 雛形の書き換える場所 (API の TemplatePlace. 種類 input は入力の形の軸, param はノードの引数, shape はノードの引数 shape の要素) */
export interface TemplatePlace {
  kind: string
  owner: string
  key: number | string
  /** 倍率 (値 × 倍率を書く) */
  scale: number
}

/** 雛形のままの定義の雛形と, 書き換える引数ごとの今の値と場所 (API の TemplateMatch) */
export interface TemplateMatch {
  name: string
  args: { name: string; value: number; places: TemplatePlace[] }[]
}

/** 雛形を書き換えた結果 (spec は書き換えた定義. args は渡した引数ごとの前と後の値と場所. notes は値を決めなかった知らせ) */
export interface TemplateApplied {
  name: string
  spec: Spec
  args: { name: string; before: number; after: number; places: TemplatePlace[] }[]
  notes?: string[]
}

/** データセットから決めた雛形の引数の値と, 決めなかった引数の知らせ */
export interface TemplateValues {
  values: Record<string, number>
  notes: string[]
}

/** 雛形の, データセットの出す値の数にする引数 (入力の形の軸に書く引数は, 入力の大きさで決める) */
const OUTPUT_ARGS: Record<string, string> = {
  unet_segmentation: 'classes',
  point_cloud_predictor: 'points_out',
}

/** 雛形の, 学ぶ分位点の水準の数にする引数 */
const LEVELS_ARGS: Record<string, string> = { tabular_demand: 'levels' }

/** 雛形の, データの幅でない入力の軸の引数 (GAN の生成器の雑音の幅. データセットから決めない) */
const NOT_DATA_ARGS: Record<string, string[]> = { gan_generator: ['noise'] }

/** 雛形の, データセットの 1 行の幅 (2 次元の入力 1 つの 2 番目の大きさ) にする, 入力の軸でない引数 */
const ROW_WIDTH_ARGS: Record<string, string> = { gan_generator: 'features' }

/** データセットの出す値の数を使わない雛形と, 出力の大きさの決まり */
const OUTPUT_RULES: Record<string, string> = {
  vae: '出力 recon は入力と同じ幅で, mu と logvar は潜在の幅',
  tabular_autoencoder: '出力 recon は入力を作り直すので入力と同じ幅',
  diffusion_denoiser: '出力 noise は入力 x に加えた雑音なので入力 x と同じ幅',
  gan_discriminator: '出力 logit は本物らしさの 1 つの値',
  gan_generator: '出力 fake は作る行なので 1 行の幅',
  contrastive_encoder: '出力 embedding は埋め込みの幅で, similarity は行どうしの似ている度合い',
  policy_value: '出力 logits は行動の数 (データセットから決めない) で, value は 1 つの値',
}

/** まとめるの出力の大きさを変えない知らせ */
const GROUP_OUTPUT_NOTE =
  '出力 embedding は埋め込みの幅 (定義で決める) で, similarity は行どうしの似ている度合いの表なので, 出力の大きさは変えていません'

/** 知らせに添える, 雛形の引数の意味 (ML-ModelControl の docs/DESIGN.md の 9 節の表の記号) */
const ARG_LABELS: Record<string, string> = {
  features: '特徴の数',
  latent: '潜在の幅',
  noise: '雑音の幅',
  users: '利用者の数',
  items: '品物の数',
  observations: '観測の数',
  actions: '行動の数',
  bottleneck: '縮めた幅',
  levels: '水準の数',
  channels: 'チャンネル数',
  classes: 'クラス数 (背景を含む)',
  points_in: '入力の点の数',
  points_out: '出す点の数',
}

/** 知らせに出す引数の名前と意味 */
function argText(name: string): string {
  return ARG_LABELS[name] ? `${name} (${ARG_LABELS[name]})` : name
}

/** 入力の最後の次元に足す数 (骨組みの約束の入力の offset. 無ければ 0) */
export function inputOffset(info: DatasetInfo, name: string): number {
  const offset = info.offsets?.[name]
  return typeof offset === 'number' && Number.isInteger(offset) && offset > 0 ? offset : 0
}

/**
 * 骨組みが読む入力の形にしたデータセットの入力 (約束の入力の offset を最後の次元に足す.
 * 効果の見積もりの S-learner は, 骨組みが処置の列を入力の終わりに足すので学習に使う列の数 + 1)
 */
export function contractInputs(info: DatasetInfo): DatasetInfo {
  if (!info.inputs.some((item) => inputOffset(info, item.name) > 0)) return info
  const widened = (shape: number[], offset: number) =>
    shape.length > 1 ? [...shape.slice(0, -1), shape.at(-1)! + offset] : shape
  return {
    ...info,
    inputs: info.inputs.map((item) => {
      const offset = inputOffset(info, item.name)
      if (offset === 0) return item
      const last = item.dataset_shape?.at(-1)
      return {
        ...item,
        shape: widened(item.shape, offset),
        dataset_shape:
          item.dataset_shape && typeof last === 'number' && item.dataset_shape.length > 1
            ? [...item.dataset_shape.slice(0, -1), last + offset]
            : item.dataset_shape,
      }
    }),
  }
}

/**
 * 雛形の書き換える引数の値をデータセットから決める
 *
 * 入力の形の軸に書く引数は, 同じ名前のデータセットの入力 (定義の入力とデータセットの入力が
 * 1 つずつなら名前の違う入力) のその軸の大きさ (データセットで決まらない軸と, 次元の数が
 * 雛形の入力と違う入力からは決めない). GAN の生成器の雑音の幅 (noise) はデータの幅でないので
 * 決めず, 出す 1 行の幅 (features) をデータセットの 2 次元の入力 1 つの幅にする.
 * unet_segmentation の classes と point_cloud_predictor の points_out は出す値の数,
 * tabular_demand の levels は学ぶ分位点の水準の数. 決めない引数は入れない (今の値のまま).
 * 入力の幅は, 骨組みの約束の入力の offset を足した幅 (contractInputs. tabular_uplift_s の
 * features は学習に使う列の数 + 処置の列の 1)
 */
export function templateValues(
  match: TemplateMatch,
  spec: Spec,
  given: DatasetInfo,
): TemplateValues {
  const info = contractInputs(given)
  const values: Record<string, number> = {}
  const notes: string[] = []
  const only =
    info.inputs.length === 1 && Object.keys(spec.inputs).length === 1 ? info.inputs[0]! : null
  const positive = (size: unknown): size is number =>
    typeof size === 'number' && Number.isInteger(size) && size > 0
  for (const arg of match.args) {
    const place = arg.places.find((found) => found.kind === 'input')
    if (NOT_DATA_ARGS[match.name]?.includes(arg.name)) {
      notes.push(
        `雛形の引数 ${argText(arg.name)} はデータの幅でないので, データセットから決めずに今の ${arg.value} のままです`,
      )
      continue
    }
    if (place) {
      const item = info.inputs.find((found) => found.name === place.owner) ?? only
      const axis = Number(place.key)
      const rank = spec.inputs[place.owner]?.shape?.length
      if (item && rank !== undefined && item.shape.length !== rank) {
        notes.push(
          `データセットの入力 ${item.name} ${shapeText(item.shape)} は雛形の入力 ${place.owner} と次元の数が違うので, ` +
            `雛形の引数 ${argText(arg.name)} は今の ${arg.value} のままです`,
        )
        continue
      }
      const size = item?.shape[axis]
      if (item && item.dataset_shape?.[axis] !== null && positive(size)) values[arg.name] = size
      continue
    }
    if (ROW_WIDTH_ARGS[match.name] === arg.name) {
      const row = info.inputs.length === 1 ? info.inputs[0]! : null
      if (row && row.shape.length === 2 && positive(row.shape[1])) values[arg.name] = row.shape[1]
      else {
        notes.push(
          `データセットの入力が 2 次元の 1 つでないので, 雛形の引数 ${argText(arg.name)} は今の ${arg.value} のままです`,
        )
      }
    }
    if (OUTPUT_ARGS[match.name] === arg.name && positive(info.outputs)) {
      values[arg.name] = info.outputs
    }
    const levels = info.demand?.levels?.length
    if (LEVELS_ARGS[match.name] === arg.name && positive(levels)) values[arg.name] = levels
  }
  return { values, notes }
}

/** 定義の名前を除いて同じ定義か (雛形の書き換えに送った定義と, 入れるときの定義を照らす) */
export function sameDefinition(first: Spec, second: Spec): boolean {
  return JSON.stringify({ ...first, name: '' }) === JSON.stringify({ ...second, name: '' })
}

/** 雛形の書き換えた場所を読み上げやすい文字にする */
function placeText(place: TemplatePlace): string {
  const times = place.scale > 1 ? ` (× ${place.scale})` : ''
  const order = typeof place.key === 'number' ? `${place.key + 1} 番目` : String(place.key)
  if (place.kind === 'input') {
    return `入力 ${place.owner} の形の ${order}`
  }
  if (place.kind === 'shape') return `${place.owner} の shape の ${order}${times}`
  return `${place.owner} の ${String(place.key)}${times}`
}

/** 雛形を書き換えた知らせ (書き換えたことと, 渡した引数ごとの前と後の値と場所. 決まらなかった値と変えない出力の知らせ) */
function templateNotes(template: TemplateApplied, info: DatasetInfo): string[] {
  const notes = [
    `定義は雛形 ${template.name} のままなので, データに合わせる引数を雛形の書き換え ` +
      '(adapt_example) で, 関わるノードの引数と入力の形をそろえて書き換えました',
  ]
  const given = new Set(template.args.map((arg) => arg.name))
  for (const arg of template.args) {
    const places = arg.places.map((place) => placeText(place)).join(', ')
    if (LEVELS_ARGS[template.name] === arg.name && info.demand?.levels) {
      const levels = info.demand.levels
      const reason =
        `学ぶ分位点の水準 ${levels.join(', ')} の数 ${arg.after} (${levelsOrigin(info.demand)}. ` +
        '水準ごとに 1 つの値)'
      const output = Object.keys(template.spec.outputs).join(', ')
      notes.push(
        arg.before === arg.after
          ? `出力 ${output} の値の数は, ${reason} のままです (雛形の引数 ${arg.name}: ${places})`
          : `出力 ${output} の値の数を, ${reason} にしました ` +
              `(雛形の引数 ${arg.name} を ${arg.before} から ${arg.after} に: ${places})`,
      )
      continue
    }
    notes.push(
      arg.before === arg.after
        ? `雛形の引数 ${argText(arg.name)} は ${arg.after} のままです (${places})`
        : `雛形の引数 ${argText(arg.name)} を ${arg.before} から ${arg.after} にしました (${places})`,
    )
  }
  notes.push(...(template.notes ?? []))
  // 決まらなかった出す値の数と水準の数
  const levelsArg = LEVELS_ARGS[template.name]
  if (levelsArg && !given.has(levelsArg)) {
    notes.push(
      '学ぶ分位点の水準の数は問題が決めるので, 出力の値の数は変えていません. ' +
        '時間をまたぐ量の判断の問題から保存先を確かめてから入れ直すと, 水準の数にします',
    )
  }
  const outputArg = OUTPUT_ARGS[template.name]
  if (outputArg && !given.has(outputArg)) {
    notes.push('タスクを決めていないデータセットなので, 出力の大きさは変えていません')
  }
  // 雛形の決まりのまま変えない出力
  if (info.price) {
    notes.push(
      '出力の値の数は変えていません (売れた数の予測は 1 行に 1 つの値なので, 雛形の 1 のまま使います)',
    )
  } else if (info.rank) {
    notes.push('出力 score は利用者と品物の全ての組の点数の表なので, 出力の大きさは変えていません')
  } else if (info.group) {
    notes.push(GROUP_OUTPUT_NOTE)
  } else if (info.uplift) {
    const outputs = Object.keys(template.spec.outputs)
    notes.push(
      `出力 ${outputs.join(' と ')} の値の数は, 雛形の 1 のままです ` +
        (outputs.length > 1
          ? '(効果の見積もりの 2 つの出力は, どちらも 1 行に 1 つの値を出します)'
          : '(結果の予測は 1 行に 1 つの値を出します)'),
    )
  } else if (
    OUTPUT_RULES[template.name] &&
    typeof info.outputs === 'number' &&
    !info.table &&
    !info.anomaly
  ) {
    // 表の生成と異常の検知は, 出す値の数が入力の幅と同じで, 初めの知らせに決め方を出す
    notes.push(
      `データセットの出す値の数 ${info.outputs} は使っていません ` +
        `(雛形 ${template.name} の出力の大きさは雛形の決まりで決まります. ${OUTPUT_RULES[template.name]})`,
    )
  }
  return notes
}

/** 部品の無い定義に置く, 出す値の数を決める全結合の部品 */
const HEAD_UNIT = 'nn.Linear'

/** 形を変える部品 (最初の次元を出す値の数にする. 点群の点の数など) */
const RESHAPE_UNIT = 'op.Reshape'

/** 部品が引数を持つか (登録簿の引数の一覧で見る) */
function hasParam(units: Map<string, UnitSpec>, unit: string, name: string): boolean {
  return units.get(unit)?.params.some((param) => param.name === name) ?? false
}

/** 入力か, それを使うつなぎ先があるか */
function isUsed(spec: Spec, units: Map<string, UnitSpec>, name: string): boolean {
  return specWires(spec, units).some((wire) => wire.source === name)
}

/** 形を読み上げやすい文字にする */
function shapeText(shape: number[]): string {
  return `[${shape.join(', ')}]`
}

/**
 * 入力の形. データセットで決まらない次元 (バッチの次元を除く. 大きさの混ざる画像の高さと幅など) は,
 * 定義の入力が同じ次元の数の形を持てばその値のままにする
 */
function mergedShape(
  item: DatasetInput,
  before: InputDef | undefined,
): { shape: number[]; kept: boolean } {
  const dims = item.dataset_shape
  const current = before?.shape
  if (
    !dims ||
    !current ||
    dims.length !== item.shape.length ||
    current.length !== item.shape.length
  ) {
    return { shape: item.shape, kept: false }
  }
  let kept = false
  const shape = item.shape.map((size, index) => {
    if (index === 0 || dims[index] !== null) return size
    kept = true
    return current[index]!
  })
  return { shape, kept }
}

/**
 * データセットに無いが残す入力の最初の次元をそろえる大きさ (バッチの次元)
 *
 * 定義の入力 (0 次元の値を除く) の最初の次元が全て同じで, そのうちの 1 つ
 * 以上がデータセットの入力と同じ名前の場合だけ, 最初の次元をバッチとみなし,
 * データセットの入力の最初の次元 (例の大きさ) を返す. 他 (最初の次元が違う
 * 入力がある, データセットの入力と同じ名前の入力が無い, データセットの入力の
 * 最初の次元がそろわない) は null (固定の形の値を変えないため)
 */
function batchSize(spec: Spec, info: DatasetInfo): number | null {
  const firsts = Object.values(spec.inputs).flatMap((value) =>
    value.shape && value.shape.length > 0 ? [value.shape[0]!] : [],
  )
  const sizes = new Set(info.inputs.map((item) => item.shape[0]))
  const [size] = sizes
  if (firsts.length === 0 || new Set(firsts).size !== 1 || sizes.size !== 1) return null
  const replaced = info.inputs.some((item) => (spec.inputs[item.name]?.shape?.length ?? 0) > 0)
  return replaced && typeof size === 'number' ? size : null
}

/** 出力の最後の形を変える部品 (op.Reshape) の最初の次元と, その前の out_features を出す値の数に合わせる */
function fitReshape(
  spec: Spec,
  units: Map<string, UnitSpec>,
  name: string,
  outputs: number,
): { spec: Spec; note: string } | null {
  const node = spec.nodes[name]
  const shape = node?.params?.shape
  if (!node || node.unit !== RESHAPE_UNIT || !Array.isArray(shape) || shape.length < 1) return null
  if (!shape.every((size) => typeof size === 'number' && Number.isInteger(size) && size > 0))
    return null
  const sources = specWires(spec, units).filter((wire) => wire.target === name)
  if (sources.length !== 1) return null
  const before = spec.nodes[sources[0]!.source]
  if (!before || !hasParam(units, before.unit, 'out_features')) return null
  const rest = (shape as number[]).slice(1)
  const features = outputs * rest.reduce((product, size) => product * size, 1)
  const reshaped = [outputs, ...rest]
  return {
    spec: {
      ...spec,
      nodes: {
        ...spec.nodes,
        [sources[0]!.source]: { ...before, params: { ...before.params, out_features: features } },
        [name]: { ...node, params: { ...node.params, shape: reshaped } },
      },
    },
    note: `${sources[0]!.source} の out_features を ${features}, ${name} の shape を ${shapeText(reshaped)} にしました`,
  }
}

/** 表の生成で, 入力と出力の幅をどう決めたかの文 */
export function tableWidthNote(table: TableWidth): string {
  const from = table.problem ? `問題 ${table.problem} が表の生成` : '表の生成の問題'
  return (
    `${from}なので, 入力と出力の幅を, 学習で使う 1 行の長さ ${table.width} にしました ` +
    `(数の項目 ${table.numbers} は 1 つずつ, 分類の項目 ${table.categories} は選べる値の数の合計 ${table.values}. ` +
    `項目の数 ${table.columns} ではありません)`
  )
}

/** 並べるで, 入力 user と item の大きさをどう決めたかの文 */
export function rankSizeNote(rank: RankSize): string {
  const from = rank.problem ? `問題 ${rank.problem} が並べる` : '並べるの問題'
  const rates =
    typeof rank.valid === 'number' && typeof rank.test === 'number'
      ? `学習のプロジェクトの学習の設定の検証用 ${rank.valid} とテスト用 ${rank.test}`
      : '学習のスクリプトの既定'
  const train = rank.rows.train ?? 0
  return (
    `${from}なので, 入力 user の大きさを学習用の利用者の数 ${rank.users}, ` +
    `item の大きさを学習用の品物の数 ${rank.items} にしました ` +
    `(学習用の反応の行の数 ${train} (時刻の順に分けたもの). 分ける割合は${rates}. ` +
    '検証用とテスト用にだけ出る利用者と品物は数えません)'
  )
}

/** まとめるで, 入力 x の幅をどう決めたかの文 */
export function groupWidthNote(group: GroupWidth): string {
  const from = group.problem ? `問題 ${group.problem} がまとめる` : 'まとめるの問題'
  return (
    `${from}なので, 入力 x の幅を, 学習で使う 1 行の長さ ${group.width} にしました ` +
    `(数の列 ${group.numbers} は 1 つずつ, 種類の列 ${group.categories} は値の種類の数の和 ${group.values}. ` +
    `群を作る列の数 ${group.columns} ではありません)`
  )
}

/**
 * 異常の検知で, 入力と出力の幅をどう決めたかの文 (自己符号化器と言うのは, 問題の設定の
 * 度合いの決め方が自己符号化器の場合だけ. 問題の設定が無いか決め方が未回答なら,
 * データセットの出す値の数の決まりとして知らせる)
 */
export function anomalyWidthNote(anomaly: AnomalyWidth): string {
  const autoencoder = anomaly.score === 'autoencoder'
  const from = anomaly.problem
    ? `問題 ${anomaly.problem} が異常の検知${autoencoder ? ' (自己符号化器)' : ''}`
    : '異常の検知のデータセット'
  const reason = autoencoder
    ? '自己符号化器は入力を作り直すので, 出力を入力と同じ幅にします'
    : `${anomaly.problem ? '度合いの決め方にまだ答えていないので, ' : ''}` +
      '出す値の数はデータセットの決まり (異常の検知は学習に使う列の数) のままです'
  const target = anomaly.target ? `. 正解の列 ${anomaly.target} は数えません` : ''
  return (
    `${from}なので, 入力と出力の幅を, 学習に使う列の数 ${anomaly.columns ?? '?'} にしました ` +
    `(${reason}${target})`
  )
}

/**
 * 効果を見積もるで, 入力の幅と出力の大きさをどう決めたかの文 (入力に入れない列は, 初めての人にも
 * 分かるよう役を言葉で添える. 名前が分からない列は役だけ). offset は骨組みが入力の終わりに足す
 * 処置の列の数 (S-learner は 1, T-learner は 0)
 */
export function upliftWidthNote(uplift: UpliftWidth, offset = 0): string {
  const from = uplift.problem
    ? `問題 ${uplift.problem} が効果の見積もり`
    : '効果を見積もるのデータセット'
  const named = (role: string, name: string | null | undefined) => (name ? `${role} ${name}` : role)
  const skipped = [
    named('処置 (行動したかどうか. クーポンを配ったかなど) の列', uplift.treatment),
    named('結果 (行動の後にどうなったか. 買ったかなど) の列', uplift.outcome),
    named('行を見分ける id の列', uplift.id),
  ].join(', ')
  if (offset > 0) {
    const width = uplift.columns === null ? '?' : String(uplift.columns + offset)
    const treatment = named('処置の列', uplift.treatment)
    const rest = [
      named('結果 (行動の後にどうなったか. 買ったかなど) の列', uplift.outcome),
      named('行を見分ける id の列', uplift.id),
    ].join(', ')
    return (
      `${from}で, 処置を入力に入れて 1 つの予測で学ぶ形 (S-learner) なので, 入力の幅を, ` +
      `学習に使う列の数 ${uplift.columns ?? '?'} に${treatment} の ${offset} を足した ${width} にしました ` +
      '(学習のスクリプトが, 処置の値の 1 か 0 を入力の最後の列に足します). ' +
      `入力に入れない列: ${rest}. ` +
      '出力 (結果の予測) は 1 行に 1 つの値を出し, 効果は処置を 1 にした予測から 0 にした予測を引いて数えます'
    )
  }
  return (
    `${from}なので, 入力の幅を, 学習に使う列の数 ${uplift.columns ?? '?'} にしました. ` +
    `入力に入れない列: ${skipped} ` +
    '(効果は処置したときとしないときの結果の差として学ぶので, 処置と結果は入力に入れません). ' +
    '効果の見積もりの 2 つの出力 (処置したときの結果の予測と, しないときの結果の予測) は, ' +
    'どちらも 1 行に 1 つの値を出します'
  )
}

/**
 * 値段を変える判断で, 入力の幅をどう決めたかの文 (列は, 初めての人にも分かるよう役を言葉で添える.
 * 無い原価と商品の id の列は出さない. 名前の分からない列は役だけ)
 */
export function priceWidthNote(price: PriceWidth): string {
  const from = price.problem
    ? `問題 ${price.problem} が値段を変える判断`
    : '値段を変える判断のデータセット'
  const named = (role: string, name: string | null | undefined) => (name ? `${role} ${name}` : role)
  const width =
    price.columns === null
      ? '入力の幅は, データセットの入力の幅のままにしました (商品の特徴の列と値段の列を合わせた数)'
      : `入力の幅を, 商品の特徴の列と値段の列を合わせた ${price.columns} 列にしました`
  const skipped = [
    named('売れた数 (予測する値) の列', price.units),
    ...(price.cost
      ? [named('原価 (仕入れか作るのにかかる値段. 利益の計算に使う) の列', price.cost)]
      : []),
    ...(price.product ? [named('商品の id (値段を選ぶ単位) の列', price.product)] : []),
    named('行を見分ける id の列', price.id),
  ].join(', ')
  return (
    `${from}なので, ${width}. ` +
    `値段 (商品の売り値) の列${price.price ? ` ${price.price} ` : ''}は, 入力の最後の列です. ` +
    `次の列は入力に入れません: ${skipped}`
  )
}

/**
 * 時間をまたぐ量の判断で, 入力の幅をどう決めたかの文 (列は, 初めての人にも分かるよう役を言葉で
 * 添える. 無い在庫の数の列は出さない. 名前の分からない列は役だけ)
 */
export function demandWidthNote(demand: DemandWidth): string {
  const from = demand.problem
    ? `問題 ${demand.problem} が時間をまたぐ量の判断`
    : '需要の記録のデータセット'
  const named = (role: string, name: string | null | undefined) => (name ? `${role} ${name}` : role)
  const width =
    demand.columns === null
      ? '入力の幅は, データセットの入力の幅のままにしました'
      : `入力の幅を, 学習に使う ${demand.columns} 列にしました`
  const skipped = [
    named('日ごとの売れた数 (先の期間の合計が予測する値) の列', demand.units),
    named('商品の id (発注の数を決める単位) の列', demand.product),
    named('日付 (発注を決める日) の列', demand.date),
    ...(demand.stock ? [named('在庫の数 (発注の数から引く) の列', demand.stock)] : []),
    named('行を見分ける id の列', demand.id),
  ].join(', ')
  return (
    `${from}なので, ${width} ` +
    '(商品の特徴, 過去の売れた数のずらしと移動平均, 曜日などの列). ' +
    `次の列は入力に入れません: ${skipped}`
  )
}

/** 学ぶ分位点の水準をどこから読んだかの文 */
function levelsOrigin(demand: DemandWidth): string {
  if (demand.levels_from === 'config') return '学習のプロジェクトの学習の設定の水準'
  return demand.problem ? `問題 ${demand.problem} の水準` : '問題の水準'
}

/** 時間をまたぐ量の判断の出力のつなぎ元からさかのぼる, 形を変えない部品 (雛形 tabular_demand の softplus と累積和) */
const KEEP_SHAPE_UNITS = new Set(['nn.Softplus', 'op.CumSum'])

/**
 * 時間をまたぐ量の判断の出力 quantiles の値の数 (学ぶ分位点の水準の数) を, 出力のつなぎ元から
 * 形を変えない部品をさかのぼった最初の out_features に入れる. 水準が分からなければ変えずに知らせる
 */
function fitDemandOutputs(
  spec: Spec,
  units: Map<string, UnitSpec>,
  demand: DemandWidth,
): DatasetApplied {
  const notes: string[] = []
  const outputNames = Object.keys(spec.outputs)
  const levels = demand.levels ?? null
  if (!levels || levels.length === 0) {
    notes.push(
      '学ぶ分位点の水準の数は問題が決めるので, 出力の値の数は変えていません. ' +
        '時間をまたぐ量の判断の問題から保存先を確かめてから入れ直すと, 水準の数にします',
    )
    return { spec, notes }
  }
  const size = levels.length
  const reason =
    `学ぶ分位点の水準 ${levels.join(', ')} の数 ${size} (${levelsOrigin(demand)}. ` +
    '水準ごとに 1 つの値)'
  const output = outputNames.length === 1 ? outputNames[0]! : null
  let name = output ? parseRef(spec.outputs[output]!).name : null
  const seen = new Set<string>()
  while (name && spec.nodes[name] && KEEP_SHAPE_UNITS.has(spec.nodes[name]!.unit)) {
    if (seen.has(name)) break
    seen.add(name)
    const sources = specWires(spec, units).filter((wire) => wire.target === name)
    name = sources.length === 1 ? sources[0]!.source : null
  }
  const node = name ? spec.nodes[name] : undefined
  if (!output || !name || !node || !hasParam(units, node.unit, 'out_features')) {
    notes.push(
      `出力の値の数 ${size} (学ぶ分位点の水準 ${levels.join(', ')} の数. ${levelsOrigin(demand)}) は, ` +
        '最後の部品の引数に手で入れてください',
    )
    return { spec, notes }
  }
  const before = node.params?.out_features
  if (before === size) {
    notes.push(`出力 ${output} の値の数は, ${reason} のままです (${name} の out_features)`)
    return { spec, notes }
  }
  notes.push(
    `出力 ${output} の値の数を, ${reason} にしました ` +
      `(${name} の out_features を ${String(before ?? '無し')} から ${size} に)`,
  )
  return {
    spec: {
      ...spec,
      nodes: { ...spec.nodes, [name]: { ...node, params: { ...node.params, out_features: size } } },
    },
    notes,
  }
}

/** 異常の検知の距離に基づく度合いで, 何も変えない理由の文 */
export function anomalyDistanceNote(anomaly: AnomalyWidth): string {
  const from = anomaly.problem ? `問題 ${anomaly.problem} の` : '問題の'
  return (
    `${from}度合いの決め方は距離に基づく度合い (学習しない) なので, モデルの定義は要りません. ` +
    '何も変えていません'
  )
}

/**
 * データセットの入力と出す値の数を定義に入れる
 *
 * 入力: 名前, 形, 型をデータセットに合わせる (1 つずつで名前が違えば名前を
 * 変えてつなぎ先も書き換える. 同じ名前は値の作り方と位置を残す. データセット
 * に無い入力は, つないでいなければ消し, つないでいれば残す. 残す入力の最初
 * の次元は, 定義の入力の最初の次元が全て同じならバッチとみなし, データ
 * セットの入力の最初の次元 (例の大きさ) にそろえる. データセットで
 * 決まらない次元は定義の値のまま). 入力に直につながる in_features の引数は
 * 入力の最後の次元, in_channels の引数は 3 次元以上の入力の 2 番目の次元
 * (チャンネル), num_embeddings の引数は文の語の番号の入力の語彙の大きさにする.
 * 出す値の数: 部品の無い定義で入力が 2 次元なら全結合の head を置いて出力に
 * つなぎ, 出力が 1 つでつなぎ元が out_features か out_channels を持てばそれ
 * にする. つなぎ元が op.Reshape なら最初の次元とその前の out_features にする.
 * 表の生成の問題の設定 (info.table) では, 入力の列の次元と出す値の数が
 * 符号化した行の幅なので, その決め方を初めに知らせる. 並べるの問題の設定
 * (info.rank) では, 入力 user と item の大きさが学習用の利用者と品物の数
 * なので, その決め方と除いた行の知らせを初めに知らせる (出力は全ての組の
 * 点数の表なので大きさを変えない). まとめるの問題の設定 (info.group) では,
 * 入力 x の幅が符号化した行の幅なので, その決め方を初めに知らせる (出力は埋め込みと
 * 類似度の表なので大きさを変えない). 異常の検知のデータセット (info.anomaly)
 * では, 入力と出力の幅が学習に使う列の数なので, その決め方を初めに知らせる.
 * 問題の設定の度合いの決め方が距離に基づく度合いなら, モデルの定義が要らない
 * ことを知らせて何も変えない. 効果を見積もるのデータセット (info.uplift) では,
 * 入力の幅が処置の列を除いた学習に使う列の数 (S-learner は骨組みが処置の列を足すので + 1) なので,
 * その決め方を初めに知らせ,
 * 出力が 2 つ以上 (T-learner の treated と control) なら, それぞれのつなぎ元の
 * out_features を出す値の数にする. 値段を変える判断のデータセット (info.price) では,
 * 入力の幅が特徴の列と値段の列の数なので, その決め方を初めに知らせ, 出力の値の数は
 * 変えない (売れた数の予測は常に 1 つの値で, 雛形 tabular_price の 1 のまま使うため).
 * 需要の記録のデータセット (info.demand) では, 入力の幅が学習に使う列の数なので, その決め方と
 * 水準の決め方の知らせを初めに知らせ, 出力の値の数を学ぶ分位点の水準の数にする (出力の
 * つなぎ元から, 形を変えない softplus と累積和をさかのぼった out_features. 水準が分からなければ
 * 変えない).
 * 入力の形は, 骨組みの約束の入力の offset を最後の次元に足した形にする (contractInputs).
 * 雛形のままの定義 (template. バックエンドが ML-ModelControl の adapt_example で書き換えた結果)
 * では, 書き換えた定義に入力を同じ決まりで入れ, 入力に直につながる引数と出す値の数をノードを
 * たどって決めずに, 書き換えた引数の前と後の値と場所を知らせる
 */
export function applyDatasetInputs(
  spec: Spec,
  raw: DatasetInfo,
  units: Map<string, UnitSpec>,
  template: TemplateApplied | null = null,
): DatasetApplied {
  // 入力の形は骨組みが読む形 (約束の入力の offset を最後の次元に足す)
  const given = contractInputs(raw)
  if (given.anomaly?.score === 'distance') {
    return { spec, notes: [anomalyDistanceNote(given.anomaly)] }
  }
  // 雛形のままの定義は, 雛形の入力と同じ名前のデータセットの入力だけを入れる (雛形の入力の
  // 名前は学習のスクリプトとの約束なので変えない)
  const skipped = template
    ? given.inputs.filter((item) => !(item.name in template.spec.inputs))
    : []
  const info =
    skipped.length > 0
      ? { ...given, inputs: given.inputs.filter((item) => !skipped.includes(item)) }
      : given
  const notes: string[] = info.table
    ? [tableWidthNote(info.table)]
    : info.rank
      ? [rankSizeNote(info.rank), ...(info.rank.notes ?? [])]
      : info.group
        ? [groupWidthNote(info.group)]
        : info.anomaly
          ? [anomalyWidthNote(info.anomaly)]
          : info.uplift
            ? [upliftWidthNote(info.uplift, inputOffset(info, info.inputs[0]?.name ?? ''))]
            : info.price
              ? [priceWidthNote(info.price)]
              : info.demand
                ? [demandWidthNote(info.demand), ...(info.demand.notes ?? [])]
                : []
  let next = spec
  // ノードと同じ名前の入力は作らない (つなぎ先が入力を指すように変わるため)
  const clashes = info.inputs.filter((item) => item.name in spec.nodes)
  if (clashes.length > 0) {
    const listed = clashes.map((item) => item.name).join(', ')
    notes.push(
      `ノード ${listed} がデータセットの入力と同じ名前なので, 何も変えていません. ノードの名前を変えてから入れ直してください`,
    )
    return { spec, notes }
  }
  const names = info.inputs.map((item) => item.name)
  // 雛形のままの定義は, ML-ModelControl で書き換えた定義から始める
  if (template) {
    next = template.spec
    for (const item of skipped) {
      notes.push(
        `データセットの入力 ${item.name} は雛形 ${template.name} の入力 ` +
          `(${Object.keys(template.spec.inputs).join(', ')}) と名前が違うので, 定義に入れていません ` +
          '(雛形の入力の名前は学習のスクリプトとの約束なので変えません)',
      )
    }
  }

  // 1 つずつで名前が違えば, 名前を変える (つなぎ先も書き換える)
  const current = Object.keys(next.inputs)
  const only = info.inputs.length === 1 ? info.inputs[0]! : null
  if (!template && only && current.length === 1 && current[0] !== only.name) {
    next = renameItem(next, current[0]!, only.name)
    notes.push(`入力 ${current[0]} の名前を ${only.name} にしました`)
  }

  // データセットに無い入力は, つないでいなければ消す. 残す入力の最初の
  // 次元は, バッチとみなせればデータセットの入力にそろえる
  const batch = batchSize(next, info)
  const inputs: Record<string, InputDef> = {}
  for (const [name, value] of Object.entries(next.inputs)) {
    if (names.includes(name)) continue
    if (isUsed(next, units, name)) {
      const shape = value.shape
      if (batch !== null && shape && shape.length > 0 && shape[0] !== batch) {
        const aligned = [batch, ...shape.slice(1)]
        inputs[name] = { ...value, shape: aligned }
        notes.push(
          `入力 ${name} はデータセットに無いですが, つないでいるので残し, ` +
            `最初の次元 (バッチ) を例の大きさ ${batch} にしました (${shapeText(aligned)})`,
        )
        continue
      }
      inputs[name] = value
      notes.push(`入力 ${name} はデータセットに無いですが, つないでいるので残しました`)
    } else {
      notes.push(`データセットに無い入力 ${name} を消しました`)
    }
  }

  // データセットの入力の形と型 (値の作り方と位置は残す)
  const shapes = new Map<string, number[]>()
  info.inputs.forEach((item, index) => {
    const before = next.inputs[item.name]
    const { shape, kept } = mergedShape(item, before)
    shapes.set(item.name, shape)
    if (kept) {
      notes.push(
        `入力 ${item.name} のデータセットで決まらない次元は, 定義の値のままにしました (${shapeText(shape)})`,
      )
    }
    inputs[item.name] = {
      ...before,
      shape,
      dtype: item.dtype,
      ui: before?.ui ?? { x: 40, y: 160 + index * 120 },
    }
  })
  next = { ...next, inputs }

  // 雛形のままの定義は, ノードをたどって引数を決めない (ML-ModelControl で書き換え済み)
  if (template) return { spec: next, notes: [...notes, ...templateNotes(template, given)] }

  // 入力に直につながる in_features を入力の最後の次元に, in_channels を
  // チャンネル (3 次元以上の入力の 2 番目の次元) に, num_embeddings を
  // 文の語彙の大きさ (埋め込みの表の行の数) に
  const vocabularies = new Map(
    info.inputs.flatMap((item) =>
      typeof item.vocabulary === 'number' && item.vocabulary > 0
        ? [[item.name, item.vocabulary] as const]
        : [],
    ),
  )
  const nodes = { ...next.nodes }
  for (const wire of specWires(next, units)) {
    const shape = shapes.get(wire.source)
    const node = nodes[wire.target]
    if (!shape || !node || shape.length < 2) continue
    const fits: [string, number][] = [['in_features', shape.at(-1)!]]
    if (shape.length >= 3) fits.push(['in_channels', shape[1]!])
    const vocabulary = vocabularies.get(wire.source)
    if (vocabulary !== undefined) fits.push(['num_embeddings', vocabulary])
    for (const [param, size] of fits) {
      const current = nodes[wire.target]!
      if (!hasParam(units, current.unit, param) || current.params?.[param] === size) continue
      nodes[wire.target] = { ...current, params: { ...current.params, [param]: size } }
      notes.push(`${wire.target} の ${param} を ${size} にしました`)
    }
  }
  next = { ...next, nodes }

  // 時間をまたぐ量の判断の出力の値の数 (学ぶ分位点の水準の数. ML-EDA の outputs の 1 は水準
  // 1 つあたりなので使わない. 部品の無い定義は, 水準が分かれば下の全結合の head の値の数にする)
  if (info.demand && (Object.keys(next.outputs).length > 0 || !info.demand.levels?.length)) {
    const fitted = fitDemandOutputs(next, units, info.demand)
    return { spec: fitted.spec, notes: [...notes, ...fitted.notes] }
  }
  // 出す値の数
  const outputs = info.outputs ?? null
  if (outputs === null) {
    notes.push(
      info.rank
        ? '出力 score は利用者と品物の全ての組の点数の表なので, 出力の大きさは変えていません'
        : info.group
          ? GROUP_OUTPUT_NOTE
          : 'タスクを決めていないデータセットなので, 出力の大きさは変えていません',
    )
    return { spec: next, notes }
  }
  const first = info.inputs[0]!
  const outputNames = Object.keys(next.outputs)
  if (Object.keys(next.nodes).length === 0 && outputNames.length === 0) {
    // 部品の無い定義は, 2 次元の入力に全結合の head を置いて出力 out につなぐ
    if (first.shape.length === 2 && units.get(HEAD_UNIT)?.available && !('head' in next.inputs)) {
      const at = next.inputs[first.name]?.ui
      const x = (at?.x ?? 40) + 280
      const y = at?.y ?? 160
      next = {
        ...next,
        nodes: {
          head: {
            unit: HEAD_UNIT,
            params: { in_features: first.shape[1]!, out_features: outputs },
            inputs: first.name,
            ui: { x, y },
          },
        },
        outputs: { out: 'head' },
        ui: { ...next.ui, outputs: { ...next.ui?.outputs, out: { x: x + 280, y } } },
      }
      notes.push(`全結合の head (出す値 ${outputs}) を置き, 出力 out につなぎました`)
      return { spec: next, notes }
    }
    notes.push(`出力の大きさ ${outputs} は, 最後の部品の引数に入れてください`)
    return { spec: next, notes }
  }
  // 値段を変える判断の売れた数の予測の値の数 (常に 1 つなので変えない)
  if (info.price) {
    notes.push(
      '出力の値の数は変えていません (売れた数の予測は 1 行に 1 つの値なので, 雛形の 1 のまま使います)',
    )
    return { spec: next, notes }
  }
  // 効果を見積もるの 2 つ以上の出力のつなぎ元の out_features (出す値の数)
  if (info.uplift && outputNames.length > 1) {
    const nodes = { ...next.nodes }
    for (const output of outputNames) {
      const from = parseRef(next.outputs[output]!).name
      const node = nodes[from]
      if (!node || !hasParam(units, node.unit, 'out_features')) {
        notes.push(`出力 ${output} の大きさ ${outputs} は, 最後の部品の引数に手で入れてください`)
        continue
      }
      if (node.params?.out_features !== outputs) {
        nodes[from] = { ...node, params: { ...node.params, out_features: outputs } }
      }
      notes.push(`${from} の out_features を ${outputs} にしました`)
    }
    return { spec: { ...next, nodes }, notes }
  }
  const source = outputNames.length === 1 ? parseRef(next.outputs[outputNames[0]!]!).name : null
  const last = source ? next.nodes[source] : undefined
  // 最後の部品の出す数の引数 (全結合は out_features, 畳み込みは out_channels).
  // 点群の outputs は値の数でなく点の数なので, 形を変える部品 (Reshape) だけで合わせる
  const cloud = info.target_kind === 'point_cloud'
  const param = cloud
    ? undefined
    : ['out_features', 'out_channels'].find(
        (name) => source && last && hasParam(units, last.unit, name),
      )
  const reshaped = source && !param ? fitReshape(next, units, source, outputs) : null
  if (source && last && param) {
    if (last.params?.[param] !== outputs) {
      next = {
        ...next,
        nodes: {
          ...next.nodes,
          [source]: { ...last, params: { ...last.params, [param]: outputs } },
        },
      }
    }
    notes.push(`${source} の ${param} を ${outputs} にしました`)
  } else if (reshaped) {
    next = reshaped.spec
    notes.push(reshaped.note)
  } else {
    notes.push(
      cloud
        ? `正解の点の数 ${outputs} は, 最後に点の数と 3 の形にする部品 (op.Reshape) を置いて手で合わせてください`
        : `出力の大きさ ${outputs} は, 最後の部品の引数に手で入れてください`,
    )
  }
  return { spec: next, notes }
}

/** 知らせに出す入力の形 (データセットで決まらない次元は ?. バッチの次元は例の大きさ) */
function summaryShape(item: DatasetInput): string {
  const dims = item.dataset_shape
  if (!dims || dims.length !== item.shape.length) return shapeText(item.shape)
  const sizes = item.shape.map((size, index) => (index > 0 && dims[index] === null ? '?' : size))
  return `[${sizes.join(', ')}]`
}

/** 入れた結果の知らせの頭 (入力の形, タスク, 目的の列か正解の項目, クラスの数, 出す値の数) */
export function datasetSummary(info: DatasetInfo): string {
  const inputs = info.inputs.map((item) => `${item.name} ${summaryShape(item)}`).join(', ')
  const parts = [`入力 ${inputs}`]
  if (info.task) parts.push(`タスク ${info.task}`)
  if (info.target_kind) parts.push(`正解 ${info.target ?? ''} (${info.target_kind})`)
  else if (info.target) parts.push(`目的の列 ${info.target}`)
  if (info.classes) parts.push(`クラス ${info.classes.length}`)
  if (info.outputs != null) parts.push(`出す値 ${info.outputs}`)
  const from = info.rank ? '反応の表' : info.group ? '群を作る表' : 'データセット'
  // 距離に基づく度合いは何も入れないので, 読んだことだけを出す
  const done = info.anomaly?.score === 'distance' ? 'を読みました' : 'から入れました'
  return `${from}${done}: ${parts.join(', ')}`
}
