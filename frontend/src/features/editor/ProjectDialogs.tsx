import { useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ClipboardList } from 'lucide-react'
import { useEffect, useId, useState } from 'react'

import { api, ApiError, type DatasetInputs, type ProjectTarget } from '../../api/client'
import { Dialog } from '../../components/Dialog'
import { Button, ErrorBox, Spinner } from '../../components/ui'
import type { Spec } from '../../lib/graph'

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
        その記録のデータセットと保存先を下の欄に入れます
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

/**
 * データセットのディレクトリを書き, ML-EDA で読んだ入力と出す値の数を定義に入れる
 *
 * 入れた結果の知らせは, 閉じるまでダイアログに出す
 */
function DatasetSection({
  disabled,
  suggested,
  onApply,
}: {
  /** 選べない理由 (部分モデルを編集している間. 選べる場合は null) */
  disabled: string | null
  /** 学習のスクリプトが読むデータセットの場所 (問題の設定の記録か, 無ければ config の data.dataset. この欄に打つ前か, 一覧から問題の設定を選んだ場合に入れる) */
  suggested: Suggestion | null
  /** 読んだ結果を定義に入れ, 知らせる文を返す */
  onApply: (info: DatasetInputs) => string[]
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
    mutationFn: (target: string) => api.datasetInputs(target),
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
            read.mutate(path.trim(), {
              onSuccess: (info) => {
                remember(DATASET_KEY, info.path)
                setResult(onApply(info))
              },
            })
          }
        }}
      >
        <PathField
          label="データセットのディレクトリ"
          hint="ML-Fathom (ML-EDA) で書き出したディレクトリ (dataset.yaml を持つもの) の絶対のパス. 入力の名前, 形, 型と出力の大きさを定義に入れます (元に戻すで戻せます)"
          value={path}
          onChange={(value) => {
            setTouched(true)
            setPath(value)
          }}
          placeholder="/path/to/project/outputs/dataset"
        />
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
}: {
  /** 保存する定義全体 */
  spec: Spec
  /** 一覧から選んだ問題の設定 (選ぶたびに新しい値. 欄に入れて保存先を確かめる) */
  choice: { path: string } | null
  /** 書いた (書いたファイルの場所) */
  onSaved: (path: string) => void
  /** 保存先を確かめた (記録のデータセットの場所をデータセットの欄に入れるため. chosen は一覧から選んだ場合) */
  onTarget: (target: ProjectTarget, text: string, chosen: boolean) => void
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
  // 書いたファイル (確かめ直すか保存し直すまで出す)
  const [written, setWritten] = useState<string | null>(null)
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
  const resolve = useMutation({
    mutationFn: ({ text, named }: { text: string; named: string; chosen: boolean }) =>
      api.projectTarget(text, named),
    onSuccess: (value, { text, named, chosen }) => {
      remember(PROJECT_KEY, text)
      setTarget({ path: text, name: named, value })
      setWritten(null)
      onTarget(value, text, chosen)
    },
    // 確かめに失敗した場合は, 前に確かめた保存先 (と書いた知らせ) を消して
    // 誤りだけを出す (確かめ直しの知らせを重ねず, 前の保存先のまま書かない)
    onError: () => {
      setTarget(null)
      setWritten(null)
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
  const { mutate: resolveTarget } = resolve
  useEffect(() => {
    if (pending) resolveTarget({ ...pending, chosen: true })
  }, [pending, resolveTarget])
  const save = useMutation({
    mutationFn: ({ text, named, overwrite }: { text: string; named: string; overwrite: boolean }) =>
      api.saveProjectModel(text, named, spec, overwrite),
    onSuccess: (saved, { text, named }) => {
      onSaved(saved.path)
      // 書いた後は既にある扱い (続けて保存すると上書きを確かめる). 保存の
      // 間に別の場所を確かめた場合は, その場所には付けない
      markExists(text, named)
      setWritten(saved.path)
    },
    onError: (error, { text, named, overwrite }) => {
      // 確かめた後に別の所で書かれた場合も, 既にある扱いにして上書きして
      // よいかを聞く
      if (!(error instanceof ApiError) || error.status !== 409 || overwrite) return
      markExists(text, named)
      if (window.confirm(`${error.message}. 上書きしますか?`)) {
        save.mutate({ text, named, overwrite: true })
      }
    },
  })
  const requestSave = () => {
    if (!checked) return
    if (checked.exists && !window.confirm(`${checked.path} は既にあります. 上書きしますか?`)) {
      return
    }
    setWritten(null)
    save.mutate({ text: path.trim(), named: name.trim(), overwrite: checked.exists })
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
            resolve.mutate({ text: path.trim(), named: name.trim(), chosen: false })
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
      {save.error && !(save.error instanceof ApiError && save.error.status === 409) && (
        <ErrorBox error={save.error} />
      )}
      <p role="status" className="text-sm">
        {written ? `${written} に保存しました` : ''}
      </p>
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
  onApply,
  onSaved,
  onClose,
}: {
  /** 保存する定義全体 */
  spec: Spec
  /** 編集している部分モデルの名前 (本体なら null. データセットの入力は本体だけ) */
  editing: string | null
  /** データセットから読んだ結果を本体の定義に入れ, 知らせる文を返す */
  onApply: (info: DatasetInputs) => string[]
  /** プロジェクトに書いた (書いたファイルの場所) */
  onSaved: (path: string) => void
  onClose: () => void
}) {
  // 学習のスクリプトが読むデータセットの場所 (記録か config の data.dataset)
  const [suggested, setSuggested] = useState<Suggestion | null>(null)
  // 一覧から選んだ問題の設定と, 最後に保存先を確かめた場所
  const [choice, setChoice] = useState<{ path: string } | null>(null)
  const [checked, setChecked] = useState<string | null>(null)
  return (
    <Dialog title="データセットとプロジェクト" onClose={onClose}>
      <div className="space-y-6">
        <StoredProblems picked={checked} onPick={(path) => setChoice({ path })} />
        <hr className="border-line" />
        <DatasetSection
          disabled={
            editing === null
              ? null
              : `部分モデル sub.${editing} を編集している間は選べません (データセットの入力は本体の入力なので, 本体に戻ってから選んでください)`
          }
          suggested={suggested}
          onApply={onApply}
        />
        <hr className="border-line" />
        <ProjectSaveSection
          spec={spec}
          choice={choice}
          onSaved={onSaved}
          onTarget={(target, text, chosen) => {
            setChecked(text)
            setSuggested({ path: target.dataset ?? null, chosen })
          }}
        />
      </div>
    </Dialog>
  )
}
