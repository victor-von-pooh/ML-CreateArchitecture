import { useMutation } from '@tanstack/react-query'
import { useId, useState } from 'react'

import { api, ApiError, type DatasetInputs, type ProjectTarget } from '../../api/client'
import { Dialog } from '../../components/Dialog'
import { Button, ErrorBox } from '../../components/ui'
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
  /** プロジェクトの記録のデータセットの場所 (この欄に打つ前なら入れる) */
  suggested: string | null
  /** 読んだ結果を定義に入れ, 知らせる文を返す */
  onApply: (info: DatasetInputs) => string[]
}) {
  const [path, setPath] = useState(() => remembered(DATASET_KEY))
  // この欄に打ったか (打った後は記録の場所で書き換えない)
  const [touched, setTouched] = useState(false)
  const [seen, setSeen] = useState(suggested)
  if (suggested !== seen) {
    setSeen(suggested)
    // 打つ前の欄だけを記録の場所にし, 次に開いたときも同じ場所を出す
    // (打った場所は, 読んだときに覚える)
    if (suggested && !touched) {
      setPath(suggested)
      remember(DATASET_KEY, suggested)
    }
  }
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
  if (target.source === 'problem') {
    return `問題の設定${target.problem ? ` ${target.problem}` : ''} の記録の保存先`
  }
  return 'プロジェクトの既定の保存先 (configs/default/train.yaml の model.spec の既定)'
}

/**
 * プロジェクトのディレクトリか問題の設定を書き, 保存先を確かめてから定義を書く
 *
 * 既にある場合は上書きしてよいかを確かめる
 */
function ProjectSaveSection({
  spec,
  onSaved,
  onTarget,
}: {
  /** 保存する定義全体 */
  spec: Spec
  /** 書いた (書いたファイルの場所) */
  onSaved: (path: string) => void
  /** 保存先を確かめた (記録のデータセットの場所をデータセットの欄に入れるため) */
  onTarget: (target: ProjectTarget) => void
}) {
  const [path, setPath] = useState(() => remembered(PROJECT_KEY))
  // 確かめた保存先 (場所を書き換えたら確かめ直す)
  const [target, setTarget] = useState<{ path: string; value: ProjectTarget } | null>(null)
  // 書いたファイル (確かめ直すか保存し直すまで出す)
  const [written, setWritten] = useState<string | null>(null)
  const titleId = useId()
  // 確かめた場所の保存先を, 既にある扱いにする
  const markExists = (text: string) =>
    setTarget((current) =>
      current && current.path === text
        ? { ...current, value: { ...current.value, exists: true } }
        : current,
    )
  const checked = target !== null && target.path === path.trim() ? target.value : null
  const resolve = useMutation({
    mutationFn: (text: string) => api.projectTarget(text),
    onSuccess: (value, text) => {
      remember(PROJECT_KEY, text)
      setTarget({ path: text, value })
      setWritten(null)
      onTarget(value)
    },
  })
  const save = useMutation({
    mutationFn: ({ text, overwrite }: { text: string; overwrite: boolean }) =>
      api.saveProjectModel(text, spec, overwrite),
    onSuccess: (saved, { text }) => {
      onSaved(saved.path)
      // 書いた後は既にある扱い (続けて保存すると上書きを確かめる). 保存の
      // 間に別の場所を確かめた場合は, その場所には付けない
      markExists(text)
      setWritten(saved.path)
    },
    onError: (error, { text, overwrite }) => {
      // 確かめた後に別の所で書かれた場合も, 既にある扱いにして上書きして
      // よいかを聞く
      if (!(error instanceof ApiError) || error.status !== 409 || overwrite) return
      markExists(text)
      if (window.confirm(`${error.message}. 上書きしますか?`)) {
        save.mutate({ text, overwrite: true })
      }
    },
  })
  const requestSave = () => {
    if (!checked) return
    if (checked.exists && !window.confirm(`${checked.path} は既にあります. 上書きしますか?`)) {
      return
    }
    setWritten(null)
    save.mutate({ text: path.trim(), overwrite: checked.exists })
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
          if (path.trim() && !resolve.isPending) resolve.mutate(path.trim())
        }}
      >
        <PathField
          label="プロジェクトか問題の設定"
          hint="ML-Pathfinder (ML-Quest) で書き出した学習のプロジェクトのディレクトリか, 問題の設定のファイル (problem.yaml など) の絶対のパス. 定義全体を, 学習のスクリプトが読む場所に書きます"
          value={path}
          onChange={setPath}
          placeholder="/path/to/project"
        />
        <Button type="submit" disabled={!path.trim() || resolve.isPending}>
          {resolve.isPending ? '確かめています…' : '保存先を確かめる'}
        </Button>
      </form>
      {resolve.error && <ErrorBox error={resolve.error} />}
      {checked && (
        <div role="group" aria-label="保存先" className="space-y-2 text-sm">
          <p>
            保存先: <span className="font-mono break-all">{checked.path}</span>
          </p>
          <p className="text-xs text-ink-3">{targetSource(checked)}</p>
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
 * データセットから入力と出力を作る欄と, プロジェクトに保存する欄を並べたダイアログ
 *
 * プロジェクトの保存先を確かめると, その記録のデータセットの場所をデータセットの欄に入れる
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
  // プロジェクトの記録のデータセットの場所
  const [suggested, setSuggested] = useState<string | null>(null)
  return (
    <Dialog title="データセットとプロジェクト" onClose={onClose}>
      <div className="space-y-6">
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
          onSaved={onSaved}
          onTarget={(target) => setSuggested(target.dataset ?? null)}
        />
      </div>
    </Dialog>
  )
}
