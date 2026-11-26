import { useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ClipboardList, FlaskConical, LayoutTemplate } from 'lucide-react'
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'

import {
  api,
  ApiError,
  contractDetail,
  type ContractFinding,
  type DatasetInputs,
  type ExperimentList,
  type ProjectTarget,
} from '../../api/client'
import { Dialog } from '../../components/Dialog'
import { Button, ErrorBox, Spinner } from '../../components/ui'
import { templateValues, type TemplateApplied } from '../../lib/dataset'
import type { Spec } from '../../lib/graph'
import { templateAction, templateConfirm, templateReason } from '../../lib/template'

// 最後に確かめたプロジェクトとデータセットの場所を覚えるブラウザの保存場所
const PROJECT_KEY = 'ml-createarchitecture:project-path'
const DATASET_KEY = 'ml-createarchitecture:dataset-path'

/** 覚えた場所を読む (読めない場合は空) */
function remembered(key: string): string {
  try {
    return window.localStorage.getItem(key) ?? ''
  } catch {
    return ''
  }
}

/** 場所を覚える (書けない場合は覚えない) */
function remember(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value)
  } catch {
    // 覚えられなくても操作は続ける
  }
}

/**
 * データセットを入れる前の準備: 読んだときの定義 (入れる前に今の定義と照らすため), 雛形のままの
 * 定義を書き換えた結果 (雛形でなければ null) と, 雛形を照らせなかった知らせ
 */
export type DatasetTemplate = { base: Spec; applied: TemplateApplied | null; notes: string[] }

/**
 * 定義が書き換えられる雛形のままなら, データセットから決めた引数で ML-ModelControl の
 * adapt_example に書き換えさせる (雛形でないか, 距離に基づく度合いで定義が要らなければ applied は
 * null). 照らすか書き換えるのに失敗した場合も, 雛形でない定義の決め方で入れられるよう applied を
 * null にして理由を知らせる (データセットを読んだ結果を捨てないため)
 */
async function adaptToDataset(spec: Spec, info: DatasetInputs): Promise<DatasetTemplate> {
  if (info.anomaly?.score === 'distance') return { base: spec, applied: null, notes: [] }
  try {
    const match = await api.matchTemplate(spec)
    if (!match) return { base: spec, applied: null, notes: [] }
    const { values, notes } = templateValues(match, spec, info)
    const adapted = await api.adaptTemplate(spec, match.name, values)
    const applied = {
      name: adapted.name,
      spec: adapted.spec as unknown as Spec,
      args: adapted.args,
      notes,
    }
    return { base: spec, applied, notes: [] }
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error)
    return {
      base: spec,
      applied: null,
      notes: [
        `雛形のままかを確かめられなかったので, 雛形でない定義の決め方で入れました (${reason})`,
      ],
    }
  }
}

/** 場所を書く欄 (Enter で決める) */
function PathField({
  label,
  hint,
  value,
  onChange,
  placeholder,
}: {
  label: string
  hint: string
  value: string
  onChange: (value: string) => void
  placeholder: string
}) {
  const id = useId()
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-xs font-semibold text-ink-2">
        {label}
      </label>
      <input
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-describedby={`${id}-hint`}
        placeholder={placeholder}
        spellCheck={false}
        className="h-8 w-full rounded-md border border-line bg-bg px-2 font-mono text-xs"
      />
      <p id={`${id}-hint`} className="text-[11px] text-ink-3">
        {hint}
      </p>
    </div>
  )
}

/** データセットの欄に入れる場所 (chosen は一覧から問題の設定を選んだ場合) */
type Suggestion = { path: string | null; chosen: boolean }

/**
 * ML-Pathfinder の置き場にある問題の設定の一覧. 押すとその問題の設定を選ぶ.
 * 使えないもの (プロジェクトを書き出していないなど) も理由を付けて並べる
 */
function StoredProblems({
  picked,
  onPick,
}: {
  /** 選んだ問題の設定の場所 */
  picked: string | null
  onPick: (path: string) => void
}) {
  const problems = useQuery({ queryKey: ['problems'], queryFn: api.listProblems })
  const list = problems.data
  const titleId = useId()
  const reasonId = useId()
  return (
    <section aria-labelledby={titleId} className="space-y-2">
      <h3 id={titleId} className="text-sm font-semibold">
        問題の設定から選ぶ
      </h3>
      <p className="text-[11px] text-ink-3">
        ML-Pathfinder でプロジェクトを書き出した問題の設定を選ぶと,
        その記録のデータセットと保存先を下の欄に入れます. 受け持ちに合う雛形があれば,
        新しく始める場合はその雛形を開きます
      </p>
      {problems.isPending && <Spinner />}
      {problems.error && <ErrorBox error={problems.error} />}
      {list?.error && <p className="text-xs text-critical-text">{list.error}</p>}
      {list && list.problems.length === 0 && !list.error && (
        <p className="text-xs text-ink-3">
          置き場 (<span className="font-mono break-all">{list.root}</span>)
          に問題の設定がありません. 下の欄に場所を書いて選べます
        </p>
      )}
      {list && list.problems.length > 0 && (
        <ul
          aria-label="ML-Pathfinder に保存した問題の設定"
          className="max-h-48 overflow-auto rounded-lg border border-line"
        >
          {list.problems.map((item, index) => (
            <li key={item.path}>
              <button
                type="button"
                onClick={() => onPick(item.path)}
                // 使えないものは理由を出して選べなくする
                disabled={!item.usable}
                aria-pressed={picked === item.path}
                aria-describedby={item.reason ? `${reasonId}-${index}` : undefined}
                className={clsx(
                  'flex w-full items-start gap-2 px-3 py-1.5 text-left text-sm hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-70',
                  picked === item.path && 'bg-accent-soft',
                )}
              >
                <ClipboardList size={14} className="mt-0.5 shrink-0 text-accent" aria-hidden />
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium">{item.name}</span>
                  <span
                    className="block truncate font-mono text-xs text-ink-3"
                    title={item.project ?? item.path}
                  >
                    {item.project ?? item.path}
                  </span>
                  {item.reason && (
                    <span id={`${reasonId}-${index}`} className="block text-xs text-warning-text">
                      {item.reason}
                    </span>
                  )}
                  {item.usable && item.example && (
                    <span className="block text-xs text-ink-2">
                      雛形: <span className="font-mono">{item.example}</span>
                    </span>
                  )}
                  {item.usable && !item.dataset && (
                    <span className="block text-xs text-ink-2">
                      データセットの記録がありません (保存先だけ入れます)
                    </span>
                  )}
                </span>
                <span className="shrink-0 text-xs text-ink-3">
                  {item.updated_at?.slice(0, 16).replace('T', ' ')}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

// 実験の状態 (run.json) の表示 (ML-Dashboard の実験の一覧と同じ言葉)
const RUN_STATUS: Record<string, string> = {
  completed: '完了',
  running: '実行中',
  stopped: '中断',
  failed: '失敗',
}

/**
 * 学習の実験に残った定義の写し (model_spec.yaml) を選んで開く欄 (「定義を開く」のダイアログの中)
 *
 * 候補を比べて選んだ定義を, 開いた後に「プロジェクトに保存」で models/ に名前を付けて保存するため.
 * 探したプロジェクトの場所は, 「プロジェクトに保存」の欄にも入るように覚える
 */
export function ExperimentCopies({
  onOpen,
  opening,
  openError,
}: {
  /** 写しを開く (場所は探したプロジェクト, run と root は一覧の run と置き場) */
  onOpen: (path: string, run: string, root: string) => void
  /** 開いている間 (押し直しを止める) */
  opening: boolean
  /** 写しを開けなかった誤り (この欄の中に出す) */
  openError: unknown
}) {
  const [path, setPath] = useState(() => remembered(PROJECT_KEY))
  // 探した場所と結果 (場所を書き換えたら探し直す)
  const [found, setFound] = useState<{ path: string; value: ExperimentList } | null>(null)
  const titleId = useId()
  const reasonId = useId()
  const search = useMutation({
    mutationFn: (text: string) => api.listExperiments(text),
    onSuccess: (value, text) => {
      remember(PROJECT_KEY, text)
      setFound({ path: text, value })
    },
    onError: () => setFound(null),
  })
  const list = found && found.path === path.trim() ? found.value : null
  return (
    <section aria-labelledby={titleId} className="space-y-2">
      <h3 id={titleId} className="mb-1 text-xs font-semibold text-ink-2">
        学習の実験の定義の写し
      </h3>
      <form
        className="space-y-2"
        onSubmit={(event) => {
          event.preventDefault()
          if (path.trim() && !search.isPending) search.mutate(path.trim())
        }}
      >
        <PathField
          label="プロジェクトか問題の設定"
          hint="実験 (ML-Found の実験のディレクトリ) に残った, 学習に使った定義の写しを開きます. 開いた後に「データセットとプロジェクト」の「プロジェクトに保存」で, models/ の下の名前を付けて保存できます"
          value={path}
          onChange={setPath}
          placeholder="/path/to/project"
        />
        <Button type="submit" disabled={!path.trim() || search.isPending}>
          {search.isPending ? '探しています…' : '実験を探す'}
        </Button>
      </form>
      {search.error && <ErrorBox error={search.error} />}
      {found !== null && !list && path.trim() && (
        <p className="text-xs text-warning-text">場所を書き換えました. 探し直してください</p>
      )}
      {list?.notes?.map((note) => (
        <p key={note} className="text-xs text-warning-text">
          {note}
        </p>
      ))}
      {list && list.experiments.length === 0 && (
        <p className="text-xs text-ink-3">
          {list.roots.length > 0
            ? `実験の置き場 (${list.roots.join(', ')}) に実験がありません`
            : '実験の置き場がありません'}
        </p>
      )}
      {list && list.experiments.length > 0 && (
        <ul
          aria-label="プロジェクトの実験"
          className="max-h-60 overflow-auto rounded-lg border border-line"
        >
          {list.experiments.map((item, index) => (
            <li key={item.path}>
              <button
                type="button"
                onClick={() => onOpen(list.project, item.run, item.root)}
                // 写しの記録が無いか, 中身が記録と違うものは理由を出して選べなくする
                disabled={!item.usable || opening}
                aria-describedby={item.reason ? `${reasonId}-${index}` : undefined}
                className="flex w-full items-start gap-2 px-3 py-1.5 text-left text-sm hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-70"
              >
                <FlaskConical size={14} className="mt-0.5 shrink-0 text-accent" aria-hidden />
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-mono">{item.run}</span>
                  <span className="block truncate text-xs text-ink-3" title={item.spec ?? ''}>
                    定義: {item.spec ?? '記録無し'}
                    {item.status ? ` · ${RUN_STATUS[item.status] ?? '未完了'}` : ''}
                  </span>
                  {item.reason && (
                    <span id={`${reasonId}-${index}`} className="block text-xs text-warning-text">
                      {item.reason}
                    </span>
                  )}
                </span>
                <span className="shrink-0 text-xs text-ink-3">
                  {item.updated_at?.slice(0, 16).replace('T', ' ')}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {openError ? <ErrorBox error={openError} /> : null}
    </section>
  )
}

/**
 * データセットのディレクトリを書き, ML-EDA で読んだ入力と出す値の数を定義に入れる
 *
 * 入れた結果の知らせは, 閉じるまでダイアログに出す
 */
function DatasetSection({
  spec,
  disabled,
  suggested,
  problem,
  onApply,
}: {
  /** 入れる先の本体の定義 (雛形のままかを照らす) */
  spec: Spec
  /** 選べない理由 (部分モデルを編集している間. 選べる場合は null) */
  disabled: string | null
  /** 学習のスクリプトが読むデータセットの場所 (問題の設定の記録か, 無ければ config の data.dataset. この欄に打つ前か, 一覧から問題の設定を選んだ場合に入れる) */
  suggested: Suggestion | null
  /** 最後に保存先を確かめた問題の設定のファイル (表の生成なら幅を符号化した行の幅に, 並べるなら入力の大きさを学習用の利用者と品物の数に, 異常の検知なら度合いの決め方を, 効果を見積もるならデータセットの説明に無い処置の列と行の id の列の名前を, 値段を変える判断ならデータセットの説明に無い値段, 原価, 商品の id と行の id の列の名前を, 時間をまたぐ量の判断なら出力の値の数にする学ぶ分位点の水準を読むため. 無ければ null) */
  problem: string | null
  /** 読んだ結果 (と雛形のままの定義を書き換えた結果) を定義に入れ, 知らせる文を返す */
  onApply: (info: DatasetInputs, template: DatasetTemplate) => string[]
}) {
  const [path, setPath] = useState(() => remembered(DATASET_KEY))
  // この欄に打ったか (打った後は記録の場所で書き換えない)
  const [touched, setTouched] = useState(false)
  const [seen, setSeen] = useState(suggested)
  // 欄に入れた記録の場所 (覚えるのは描画の後)
  const [applied, setApplied] = useState<string | null>(null)
  if (suggested !== seen) {
    setSeen(suggested)
    // 打つ前の欄だけを記録の場所にする (打った場所は, 読んだときに覚える).
    // 一覧から選んだ場合は, 打った後でも選んだ記録の場所にする
    if (suggested?.path && (!touched || suggested.chosen)) {
      setPath(suggested.path)
      setApplied(suggested.path)
      setTouched(false)
    }
  }
  // 欄に入れた記録の場所を, 次に開いたときにも出すように覚える
  useEffect(() => {
    if (applied) remember(DATASET_KEY, applied)
  }, [applied])
  const [result, setResult] = useState<string[] | null>(null)
  const titleId = useId()
  const disabledId = useId()
  // 定義に入れるのは mutate に渡した onSuccess (読んでいる間にダイアログを
  // 閉じた場合は入れない)
  const read = useMutation({
    mutationFn: async ({ target, from }: { target: string; from: string | null }) => {
      const info = await api.datasetInputs(target, from)
      return { info, template: await adaptToDataset(spec, info) }
    },
  })
  return (
    <section aria-labelledby={titleId} className="space-y-3">
      <h3 id={titleId} className="text-sm font-semibold">
        データセットから入力と出力を作る
      </h3>
      <form
        className="space-y-3"
        onSubmit={(event) => {
          event.preventDefault()
          if (path.trim() && !read.isPending && !disabled) {
            setResult(null)
            read.mutate(
              { target: path.trim(), from: problem },
              {
                onSuccess: ({ info, template }) => {
                  remember(DATASET_KEY, info.path)
                  setResult(onApply(info, template))
                },
              },
            )
          }
        }}
      >
        <PathField
          label="データセットのディレクトリ"
          hint="ML-Fathom (ML-EDA) で書き出したディレクトリ (dataset.yaml を持つもの) の絶対のパス. 並べるの問題の設定では, 学習のスクリプトが読む反応の表のファイル. 入力の名前, 形, 型と出力の大きさを定義に入れます (元に戻すで戻せます)"
          value={path}
          onChange={(value) => {
            setTouched(true)
            setPath(value)
          }}
          placeholder="/path/to/project/outputs/dataset"
        />
        {problem && (
          <p className="text-[11px] text-ink-3">
            問題の設定 <span className="font-mono break-all">{problem}</span> も渡します.
            表の生成なら, 入力と出力の幅を, 学習で使う 1 行の長さにします (数の項目は 1 つずつ,
            分類の項目は選べる値の数の合計). 並べるなら, 入力 user と item の大きさを,
            反応を時刻の順に分けた学習用の反応にいる利用者と品物の数にします. 異常の検知で,
            度合いの決め方が距離に基づく度合いなら, モデルの定義が要らないので何も変えません.
            効果を見積もるなら, 入力に入れない処置 (行動したかどうか) の列と行を見分ける id
            の列の名前を, データセットに書いていなければ問題の設定から読んで知らせます.
            値段を変える判断なら, 入力の最後に入る値段 (商品の売り値) の列と, 入力に入れない
            売れた数 (予測する値), 原価 (仕入れか作るのにかかる値段), 商品の id (値段を選ぶ単位)
            と行を見分ける id の列を知らせます (列の名前は, データセットに書いていなければ問題の
            設定から読みます). 時間をまたぐ量の判断なら, 出力の値の数を, 学ぶ分位点の水準の数
            (プロジェクトの学習の設定にあればそれ, 無ければ問題の設定の水準) にします
          </p>
        )}
        <Button
          type="submit"
          variant="primary"
          disabled={!path.trim() || read.isPending || disabled !== null}
          aria-describedby={disabled ? disabledId : undefined}
        >
          {read.isPending ? '読んでいます…' : '入力と出力を作る'}
        </Button>
        {disabled && (
          <p id={disabledId} className="text-xs text-warning-text">
            {disabled}
          </p>
        )}
      </form>
      {read.error && <ErrorBox error={read.error} />}
      {/* 読み上げの欄は先に置いておき, 結果を入れる */}
      <div role="status" className="text-sm">
        {result && (
          <ul className="list-disc space-y-1 pl-5">
            {result.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}

/** 問題の設定の受け持ちに合う雛形 (opened は開いたか) */
export type Template = {
  example: string
  problem: string | null
  opened: boolean
  reason: string | null
}

/**
 * 問題の設定の受け持ちに合う雛形 (ML-ModelControl の同梱の例) の知らせ. 開いていなければ「雛形を開く」を出す
 */
export function TemplateNote({
  template,
  opening,
  error,
  onOpen,
}: {
  template: Template
  /** 雛形を読んでいる間 (押し直しを止める) */
  opening: boolean
  /** 雛形を読めなかった誤り */
  error: unknown
  onOpen: () => void
}) {
  const titleId = useId()
  const from = template.problem ? `問題の設定 ${template.problem} ` : '問題の設定'
  return (
    <section
      aria-labelledby={titleId}
      className="space-y-2 rounded-lg border border-line bg-surface-2 px-3 py-2"
    >
      <h3 id={titleId} className="flex items-center gap-1.5 text-sm font-semibold">
        <LayoutTemplate size={14} className="text-accent" aria-hidden />
        受け持ちに合う雛形
      </h3>
      <p role="status" className="text-xs">
        {template.opened
          ? `${from}の受け持ちに合う雛形 ${template.example} を開きました. 入力と出力はデータセットから作れます`
          : `${from}の受け持ちに合う雛形 ${template.example} があります${template.reason ? ` (${template.reason})` : ''}`}
      </p>
      {!template.opened && (
        <Button disabled={opening} onClick={onOpen}>
          {opening ? '開いています…' : `雛形 ${template.example} を開く`}
        </Button>
      )}
      {error ? <ErrorBox error={error} /> : null}
    </section>
  )
}

/** 保存先の決め方の説明 */
function targetSource(target: ProjectTarget): string {
  if (target.source === 'name') {
    return 'models/ の下の名前の保存先 (config の model.spec は変えません. ML-Dashboard の学習の起動の「モデルの定義の候補」で選べます)'
  }
  if (target.source === 'problem') {
    return `問題の設定${target.problem ? ` ${target.problem}` : ''} の記録の保存先`
  }
  return 'プロジェクトの既定の保存先 (configs/default/train.yaml の model.spec の既定)'
}

/** プロジェクトに保存する要求 (saved は送る定義, accept は骨組みの約束に合わなくても書くか) */
type SaveRequest = {
  text: string
  named: string
  checkedPath: string
  overwrite: boolean
  saved: Spec
  accept: boolean
  /** 送ったときの確かめの番号 (保存の間に確かめ直したかを見分ける) */
  resolved: number
}

/**
 * 骨組みの学習のスクリプトが読む入力と出力に合わない所 (直す場所と文) と, 直さずに保存するボタン
 */
function ContractFindings({
  contract,
  pending,
  onAccept,
}: {
  contract: { message: string; findings: ContractFinding[]; notes: string[] }
  pending: boolean
  onAccept: () => void
}) {
  const titleId = useId()
  return (
    <section
      aria-labelledby={titleId}
      className="space-y-2 rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning-text"
    >
      <h4 id={titleId} className="font-semibold">
        骨組みの入力と出力に合わない所
      </h4>
      <p role="alert">保存しませんでした: {contract.message}</p>
      <ul className="list-disc space-y-1 pl-5">
        {contract.findings.map((item) => (
          <li key={`${item.id}:${item.message}`}>
            <span className="font-mono">{item.id}</span>: {item.message}
          </li>
        ))}
      </ul>
      {contract.notes.map((note, index) => (
        <p key={index}>{note}</p>
      ))}
      <Button disabled={pending} onClick={onAccept}>
        直さずに保存
      </Button>
    </section>
  )
}

/**
 * プロジェクトのディレクトリか問題の設定を書き, 保存先を確かめてから定義を書く
 *
 * models/ の下の名前を書くと models/<名前>.yaml に書く (空なら既定の保存先). 既にある場合は上書きしてよいかを確かめる
 */
function ProjectSaveSection({
  spec,
  choice,
  onSaved,
  onTarget,
  onFailed,
}: {
  /** 保存する定義全体 */
  spec: Spec
  /** 一覧から選んだ問題の設定 (選ぶたびに新しい値. 欄に入れて保存先を確かめる) */
  choice: { path: string } | null
  /** 書いた (書いたファイルの場所) */
  onSaved: (path: string) => void
  /** 保存先を確かめた (記録のデータセットの場所をデータセットの欄に入れるため. chosen は一覧から選んだ場合) */
  onTarget: (target: ProjectTarget, text: string, chosen: boolean) => void
  /** 保存先の確かめに失敗した (確かめた場所とデータセットの欄の候補を消すため) */
  onFailed: () => void
}) {
  const [path, setPath] = useState(() => remembered(PROJECT_KEY))
  // models/ の下の名前 (候補ごとに変えるので覚えない. 空なら既定の保存先)
  const [name, setName] = useState('')
  // 確かめた保存先 (場所か名前を書き換えたら確かめ直す)
  const [target, setTarget] = useState<{
    path: string
    name: string
    value: ProjectTarget
  } | null>(null)
  // 書いたファイルと確かめの知らせ (確かめ直すか保存し直すまで出す)
  const [written, setWritten] = useState<{ path: string; notes: string[] } | null>(null)
  // 骨組みの約束に合わない所で断られた保存 (定義と保存先が同じ間だけ出し, 直さずに保存できる)
  // (定義は中身で比べる. 親が描くたびに新しい値を渡すため)
  const [contract, setContract] = useState<{
    text: string
    request: SaveRequest
    message: string
    findings: ContractFinding[]
    notes: string[]
  } | null>(null)
  // 保存が 409 で確かめ直した理由 (次に確かめるまで出す)
  const [conflict, setConflict] = useState<string | null>(null)
  const titleId = useId()
  // 確かめた場所の保存先を, 既にある扱いにする
  const markExists = (text: string, named: string) =>
    setTarget((current) =>
      current && current.path === text && current.name === named
        ? { ...current, value: { ...current.value, exists: true } }
        : current,
    )
  const checked =
    target !== null && target.path === path.trim() && target.name === name.trim()
      ? target.value
      : null
  // 最後に送った確かめの番号 (確かめが重なった場合に, 遅れて返った前の
  // 要求の結果で後の結果を上書きしないため. 送るのは resolveTarget だけ)
  const latest = useRef(0)
  const resolve = useMutation({
    mutationFn: ({
      text,
      named,
    }: {
      text: string
      named: string
      chosen: boolean
      id: number
      /** 保存の 409 で確かめ直す場合の理由 */
      reason?: string
    }) => api.projectTarget(text, named),
    onSuccess: (value, { text, named, chosen, id, reason }) => {
      if (id !== latest.current) return
      remember(PROJECT_KEY, text)
      setTarget({ path: text, name: named, value })
      setWritten(null)
      setContract(null)
      setConflict(reason ?? null)
      onTarget(value, text, chosen)
    },
    // 確かめに失敗した場合は, 前に確かめた保存先 (と書いた知らせ) を消して
    // 誤りだけを出す (確かめ直しの知らせを重ねず, 前の保存先のまま書かない)
    onError: (_error, { id }) => {
      if (id !== latest.current) return
      setTarget(null)
      setWritten(null)
      setContract(null)
      setConflict(null)
      onFailed()
    },
  })
  // 一覧から選んだ問題の設定を欄に入れ, 今の名前のまま保存先を確かめる
  // (欄は描画の中で, 確かめる要求は描画の後)
  const [seenChoice, setSeenChoice] = useState(choice)
  const [pending, setPending] = useState<{ text: string; named: string } | null>(null)
  if (choice !== seenChoice) {
    setSeenChoice(choice)
    if (choice) {
      setPath(choice.path)
      setPending({ text: choice.path, named: name.trim() })
    }
  }
  const { mutate: resolveMutate } = resolve
  const resolveTarget = useCallback(
    (request: { text: string; named: string; chosen: boolean; reason?: string }) => {
      latest.current += 1
      resolveMutate({ ...request, id: latest.current })
    },
    [resolveMutate],
  )
  useEffect(() => {
    if (pending) resolveTarget({ ...pending, chosen: true })
  }, [pending, resolveTarget])
  const save = useMutation({
    mutationFn: ({ text, named, checkedPath, overwrite, saved, accept }: SaveRequest) =>
      api.saveProjectModel(text, named, saved, checkedPath, overwrite, accept),
    onSuccess: (saved, { text, named }) => {
      onSaved(saved.path)
      // 書いた後は既にある扱い (続けて保存すると上書きを確かめる). 保存の
      // 間に別の場所を確かめた場合は, その場所には付けない
      markExists(text, named)
      setWritten({ path: saved.path, notes: saved.notes ?? [] })
    },
    onError: (error, request) => {
      // 骨組みの約束に合わない所は, 直す場所を出して書くかを任せる
      const found = contractDetail(error)
      if (found) {
        // 保存の間に保存先を確かめ直した場合は出さない (古い要求の結果のため)
        if (request.resolved !== latest.current) return
        setContract({ text: JSON.stringify(request.saved), request, ...found })
        return
      }
      // 確かめた後に保存先が変わったか, 別の所でファイルが作られた (409)
      // 場合は, 保存先を確かめ直して理由を出す (確かめていない保存先と,
      // 既にあると知らせていないファイルに書かないため)
      if (!(error instanceof ApiError) || error.status !== 409) return
      const { text, named } = request
      resolveTarget({ text, named, chosen: false, reason: error.message })
    },
  })
  // 断られた保存は, 同じ定義と同じ保存先の間だけ出す (定義を直したら消す)
  const specText = useMemo(() => JSON.stringify(spec), [spec])
  const shownContract =
    contract !== null &&
    contract.text === specText &&
    checked?.path === contract.request.checkedPath
      ? contract
      : null
  // 断られた後 (保存の間も含む) に定義を書き換えた場合は, 合わない所の代わりに保存し直す案内
  const staleContract =
    contract !== null && shownContract === null && checked?.path === contract.request.checkedPath
  // 「直さずに保存」で欄が消えた後の focus の移り先 (ダイアログの外に落とさないため)
  const statusRef = useRef<HTMLParagraphElement>(null)
  const requestSave = () => {
    if (!checked) return
    if (checked.exists && !window.confirm(`${checked.path} は既にあります. 上書きしますか?`)) {
      return
    }
    setWritten(null)
    setContract(null)
    setConflict(null)
    save.mutate({
      text: path.trim(),
      named: name.trim(),
      checkedPath: checked.path,
      overwrite: checked.exists,
      saved: spec,
      accept: false,
      resolved: latest.current,
    })
  }
  // 合わない所を残したまま, 断られた保存と同じ保存先に書く
  const acceptSave = () => {
    if (!shownContract) return
    setWritten(null)
    setContract(null)
    save.mutate({ ...shownContract.request, accept: true, resolved: latest.current })
    statusRef.current?.focus()
  }
  return (
    <section aria-labelledby={titleId} className="space-y-3">
      <h3 id={titleId} className="text-sm font-semibold">
        プロジェクトに保存
      </h3>
      <form
        className="space-y-3"
        onSubmit={(event) => {
          event.preventDefault()
          if (path.trim() && !resolve.isPending) {
            resolveTarget({ text: path.trim(), named: name.trim(), chosen: false })
          }
        }}
      >
        <PathField
          label="プロジェクトか問題の設定"
          hint="ML-Pathfinder (ML-Quest) で書き出した学習のプロジェクトのディレクトリか, 問題の設定のファイル (problem.yaml など) の絶対のパス. 定義全体を, 学習のスクリプトが読む場所 (下の名前を書いた場合は, このプロジェクトの models/<名前>.yaml) に書きます"
          value={path}
          onChange={setPath}
          placeholder="/path/to/project"
        />
        <PathField
          label="models/ の下の名前"
          hint="モデルの候補を並べる場合に, プロジェクトの models/<名前>.yaml に書きます (英数字, _, -. 先頭に - は使えない. 64 文字まで. 拡張子は書かない. 同じ名前の .yml があればそこに書く). 空なら上の既定の保存先. config の model.spec は変えず, ML-Dashboard の学習の起動の「モデルの定義の候補」で選びます"
          value={name}
          onChange={setName}
          placeholder="空なら既定の保存先 (例: mlp)"
        />
        <Button type="submit" disabled={!path.trim() || resolve.isPending}>
          {resolve.isPending ? '確かめています…' : '保存先を確かめる'}
        </Button>
      </form>
      {resolve.error && <ErrorBox error={resolve.error} />}
      {target !== null && !checked && path.trim() && (
        <p className="text-xs text-warning-text">
          場所か名前を書き換えました. 保存先を確かめ直してください
        </p>
      )}
      {checked && (
        <div role="group" aria-label="保存先" className="space-y-2 text-sm">
          <p>
            保存先: <span className="font-mono break-all">{checked.path}</span>
          </p>
          <p className="text-xs text-ink-3">{targetSource(checked)}</p>
          {checked.source === 'name' && checked.default && (
            <p className="rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning-text">
              既定の保存先 (記録か config の model.spec) と同じファイルです.
              保存すると既定の定義を上書きします
            </p>
          )}
          {conflict && (
            <p
              role="alert"
              className="rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning-text"
            >
              保存しませんでした: {conflict}. 保存先を確かめ直したので,
              見直してからもう一度保存してください
            </p>
          )}
          {checked.exists && (
            <p className="rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning-text">
              既にファイルがあります. 保存すると上書きします
            </p>
          )}
          <Button variant="primary" disabled={save.isPending} onClick={requestSave}>
            {save.isPending ? '保存しています…' : 'この場所に保存'}
          </Button>
        </div>
      )}
      {shownContract && (
        <ContractFindings contract={shownContract} pending={save.isPending} onAccept={acceptSave} />
      )}
      {staleContract && (
        <p
          role="alert"
          className="rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning-text"
        >
          保存しませんでした: 骨組みの入力と出力に合わない所がありました.
          その後に定義を書き換えたので, もう一度保存してください
        </p>
      )}
      {save.error && !(save.error instanceof ApiError && save.error.status === 409) && (
        <ErrorBox error={save.error} />
      )}
      <p ref={statusRef} tabIndex={-1} role="status" className="text-sm focus:outline-none">
        {written ? `${written.path} に保存しました` : ''}
      </p>
      {written && written.notes.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-xs text-ink-3">
          {written.notes.map((note, index) => (
            <li key={index}>{note}</li>
          ))}
        </ul>
      )}
    </section>
  )
}

/**
 * 問題の設定の一覧, データセットから入力と出力を作る欄, プロジェクトに保存する欄を並べたダイアログ
 *
 * プロジェクトの保存先を確かめると, その記録のデータセットの場所をデータセットの欄に入れる.
 * 一覧から問題の設定を選ぶと, 保存先を確かめ, データセットの欄も記録の場所にする
 */
export function ProjectDialog({
  spec,
  editing,
  fresh,
  dirty,
  onApply,
  onSaved,
  onTemplate,
  onClose,
}: {
  /** 保存する定義全体 */
  spec: Spec
  /** 編集している部分モデルの名前 (本体なら null. データセットの入力は本体だけ) */
  editing: string | null
  /** 今の定義が新しく始める定義か (雛形をそのまま開いてよいか) */
  fresh: boolean
  /** 保存していない変更があるか (雛形を開く前に確かめる) */
  dirty: boolean
  /** データセットから読んだ結果 (と雛形のままの定義を書き換えた結果) を本体の定義に入れ, 知らせる文を返す */
  onApply: (info: DatasetInputs, template: DatasetTemplate) => string[]
  /** プロジェクトに書いた (書いたファイルの場所) */
  onSaved: (path: string) => void
  /** 問題の設定の雛形の定義を開く (開いた知らせはこのダイアログに出す) */
  onTemplate: (spec: Spec) => void
  onClose: () => void
}) {
  // 学習のスクリプトが読むデータセットの場所 (記録か config の data.dataset)
  const [suggested, setSuggested] = useState<Suggestion | null>(null)
  // 一覧から選んだ問題の設定と, 最後に保存先を確かめた場所
  const [choice, setChoice] = useState<{ path: string } | null>(null)
  const [checked, setChecked] = useState<string | null>(null)
  // 最後に確かめた保存先の問題の設定のファイル (データセットの欄に渡す)
  const [problemPath, setProblemPath] = useState<string | null>(null)
  // 最後に確かめた問題の設定の雛形 (無ければ null)
  const [template, setTemplate] = useState<Template | null>(null)
  // 最後に送った雛形の読み込みの番号 (読み込みの間に別の問題の設定を確かめた
  // 場合に, 前の雛形を開かないため)
  const latestTemplate = useRef(0)
  const load = useMutation({
    mutationFn: ({ example }: { example: string; problem: string | null; id: number }) =>
      api.getExample(example),
    onSuccess: (model, { example, problem, id }) => {
      if (id !== latestTemplate.current) return
      onTemplate(model.spec as unknown as Spec)
      setTemplate({ example, problem, opened: true, reason: null })
    },
  })
  const openTemplate = (example: string, problem: string | null) => {
    latestTemplate.current += 1
    load.mutate({ example, problem, id: latestTemplate.current })
  }
  return (
    <Dialog title="データセットとプロジェクト" onClose={onClose}>
      <div className="space-y-6">
        <StoredProblems picked={checked} onPick={(path) => setChoice({ path })} />
        {template && (
          <TemplateNote
            template={template}
            opening={load.isPending}
            error={load.error}
            onOpen={() => {
              const question = templateConfirm({ fresh, dirty })
              if (question && !window.confirm(question)) return
              openTemplate(template.example, template.problem)
            }}
          />
        )}
        <hr className="border-line" />
        <DatasetSection
          spec={spec}
          disabled={
            editing === null
              ? null
              : `部分モデル sub.${editing} を編集している間は選べません (データセットの入力は本体の入力なので, 本体に戻ってから選んでください)`
          }
          suggested={suggested}
          problem={problemPath}
          onApply={onApply}
        />
        <hr className="border-line" />
        <ProjectSaveSection
          spec={spec}
          choice={choice}
          onSaved={onSaved}
          onTarget={(target, text, chosen) => {
            setChecked(text)
            setProblemPath(target.problem_path ?? null)
            setSuggested({ path: target.dataset ?? null, chosen })
            // 雛形は, 新しく始める場合だけ開く (今の定義も保存先の定義も上書きしない)
            const example = target.example ?? null
            const action = templateAction({ example, chosen, fresh, exists: target.exists })
            latestTemplate.current += 1
            load.reset()
            if (!example || action === null) {
              setTemplate(null)
              return
            }
            const problem = target.problem ?? null
            const reason = templateReason({ fresh, dirty, exists: target.exists })
            setTemplate({ example, problem, opened: false, reason })
            if (action === 'open') openTemplate(example, problem)
          }}
          onFailed={() => {
            setChecked(null)
            setProblemPath(null)
            setSuggested(null)
            latestTemplate.current += 1
            setTemplate(null)
          }}
        />
      </div>
    </Dialog>
  )
}
