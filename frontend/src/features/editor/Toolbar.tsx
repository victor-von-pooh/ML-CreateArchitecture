import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  Code2,
  FileText,
  FolderOpen,
  FolderOutput,
  LogIn,
  LogOut,
  Moon,
  Network,
  Redo2,
  Save,
  Sun,
  Trash2,
  Undo2,
} from 'lucide-react'
import { useEffect, useId, useState } from 'react'

import { useApi, type CheckResult, type DatasetInputs } from '../../api/client'
import { Dialog } from '../../components/Dialog'
import { Button, ErrorBox, ExampleButton, Spinner } from '../../components/ui'
import { failedNodes, formatCount } from '../../lib/format'
import { danglingNodes, emptySpec, freshName, type Spec, type UnitSpec } from '../../lib/graph'
import { subgraphCheckTarget } from '../../lib/subgraph'
import { useStage } from '../../lib/stage'
import { useTheme } from '../../lib/themeContext'
import { startsFresh } from '../../lib/template'
import { ExperimentCopies, ProjectDialog, type DatasetTemplate } from './ProjectDialogs'

/** 文字の内容を表示し, コピーできる枠 */
function TextView({ text, fileName }: { text: string; fileName?: string }) {
  // コピーした結果 (クリップボードを使えない場合は理由)
  const [copied, setCopied] = useState<'done' | 'failed' | null>(null)
  // ファイルとして保存 (ブラウザのダウンロード)
  const download = () => {
    if (!fileName) return
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url
    link.download = fileName
    link.click()
    URL.revokeObjectURL(url)
  }
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <Button
          variant="secondary"
          onClick={() => {
            void navigator.clipboard.writeText(text).then(
              () => setCopied('done'),
              () => setCopied('failed'),
            )
          }}
        >
          {copied === 'done'
            ? 'コピーしました'
            : copied === 'failed'
              ? 'コピーできません (下の文字を選んでコピーしてください)'
              : 'コピー'}
        </Button>
        {fileName && text && (
          <Button variant="secondary" onClick={download}>
            {fileName} をダウンロード
          </Button>
        )}
      </div>
      <pre className="overflow-auto rounded-md bg-surface-2 p-3 font-mono text-xs">{text}</pre>
    </div>
  )
}

/** 保存した定義か, 同梱した例を選んで開く */
function OpenDialog({
  onOpen,
  onOpenExperiment,
  opening,
  experimentError,
  onClose,
  onDeleted,
  openError,
}: {
  /** 開けなかった誤り (読めない YAML など. ダイアログの中に出す) */
  openError: unknown
  onOpen: (source: 'model' | 'example', name: string) => void
  /** 実験の定義の写しを開く (path はプロジェクト, run は実験, root は置き場) */
  onOpenExperiment: (path: string, run: string, root: string) => void
  /** 開いている間 */
  opening: boolean
  /** 実験の写しを開けなかった誤り (実験の欄の中に出す) */
  experimentError: unknown
  onClose: () => void
  /** 保存した定義を消した後 (今開いている定義なら, 保存していない扱いにする) */
  onDeleted: (name: string) => void
}) {
  const api = useApi()
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })
  const queryClient = useQueryClient()
  // 保存した定義を消す (確かめてから. 消した後は一覧を読み直す)
  const remove = useMutation({
    mutationFn: (name: string) => api.deleteModel(name),
    onSuccess: (_, name) => onDeleted(name),
    onSettled: () => void queryClient.invalidateQueries({ queryKey: ['models'] }),
  })
  const examples = useQuery({
    queryKey: ['examples'],
    queryFn: api.listExamples,
    staleTime: Infinity,
  })
  return (
    <Dialog title="定義を開く" onClose={onClose}>
      <h3 className="mb-1 text-xs font-semibold text-ink-2">保存した定義</h3>
      {models.isPending ? (
        <Spinner />
      ) : models.error ? (
        <ErrorBox error={models.error} />
      ) : models.data.length === 0 ? (
        <p className="mb-3 text-sm text-ink-3">保存した定義はまだありません.</p>
      ) : (
        <ul className="mb-4 divide-y divide-line">
          {models.data.map((model) => (
            <li key={model.name} className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => onOpen('model', model.name)}
                className="flex min-w-0 flex-1 items-center justify-between px-2 py-2 text-left hover:bg-surface-2"
              >
                <span className="font-mono text-sm">{model.name}</span>
                <span className="text-xs text-ink-3">{model.updated_at.replace('T', ' ')}</span>
              </button>
              <Button
                variant="ghost"
                icon={Trash2}
                disabled={remove.isPending}
                onClick={() => {
                  if (window.confirm(`保存した定義 ${model.name} を消しますか? 元に戻せません.`)) {
                    remove.mutate(model.name)
                  }
                }}
                aria-label={`${model.name} を消す`}
                title={`${model.name} を消す`}
              />
            </li>
          ))}
        </ul>
      )}
      {remove.error && <ErrorBox error={remove.error} />}
      {openError ? <ErrorBox error={openError} /> : null}
      <h3 className="mb-1 text-xs font-semibold text-ink-2">同梱の例から始める</h3>
      {examples.isPending ? (
        <Spinner />
      ) : examples.error ? (
        <ErrorBox error={examples.error} />
      ) : (
        <ul className="divide-y divide-line">
          {examples.data.map((example) => (
            <li key={example.name}>
              <ExampleButton
                name={example.name}
                description={example.description}
                onClick={() => onOpen('example', example.name)}
              />
            </li>
          ))}
        </ul>
      )}
      <hr className="my-4 border-line" />
      <ExperimentCopies onOpen={onOpenExperiment} opening={opening} openError={experimentError} />
    </Dialog>
  )
}

/** 画面の上の操作 (新規, 開く, 保存, 入力と出力の追加, YAML とコードの表示) */
export function Toolbar({
  savedNotice,
  spec,
  units,
  whole,
  editing,
  fileName,
  dirty,
  check,
  checking,
  onChange,
  onLoad,
  onSaved,
  onSelect,
  center,
  onUndo,
  onRedo,
  onArrange,
  ownFile,
  onDeletedOwn,
  initialName,
  onNameChange,
  onSavingChange,
  onDataset,
  onProjectSaved,
  onTemplate,
}: {
  /** 保存した知らせ (保存すると上の操作を作り直すので, 外で持つ) */
  savedNotice: string
  /** 画面で編集している定義 (部分モデルを編集している間はその部分モデル) */
  spec: Spec
  /** 部品の一覧 (出力のつなぎ先の口の数を見るため) */
  units: Map<string, UnitSpec>
  /** 保存と書き出しをする定義全体 */
  whole: Spec
  /** 編集している部分モデルの名前 (本体なら null) */
  editing: string | null
  fileName: string
  dirty: boolean
  check: CheckResult | undefined
  checking: boolean
  onChange: (spec: Spec) => void
  onLoad: (name: string, spec: Spec, source: 'model' | 'example' | 'experiment' | 'new') => void
  onSaved: (name: string, spec: Spec) => void
  onSelect: (id: string) => void
  center: (
    size?: { width: number; height: number },
    near?: { x: number; y: number },
  ) => { x: number; y: number }
  /** 元に戻す (戻せない場合は undefined) */
  onUndo?: () => void
  /** やり直す (やり直せない場合は undefined) */
  onRedo?: () => void
  /** 全てをつながりの深さで段に分けて並べ直す */
  onArrange: () => void
  /** 今の定義を開いた (か保存した) 保存した定義の名前 (新規と例は null) */
  ownFile: string | null
  /** 今開いている保存した定義を消した (保存していない扱いにする) */
  onDeletedOwn: () => void
  /** 保存する名前の欄の最初の値 (下書きから戻した名前. 無ければ fileName) */
  initialName?: string
  /** 保存する名前の欄を書き換えた (下書きに残すため) */
  onNameChange?: (name: string) => void
  /** 保存しているか (保存の間は下書きを戻す操作を止めるため) */
  onSavingChange?: (saving: boolean) => void
  /** データセットから読んだ入力と出す値の数 (と雛形のままの定義を書き換えた結果) を本体の定義に入れ, 知らせる文を返す */
  onDataset: (info: DatasetInputs, template: DatasetTemplate) => string[]
  /** 定義をプロジェクトに書いた (書いたファイルの場所と, 書いた定義) */
  onProjectSaved: (path: string, spec: Spec) => void
  /** 問題の設定の受け持ちに合う雛形の定義を開く */
  onTemplate: (spec: Spec) => void
}) {
  const api = useApi()
  const queryClient = useQueryClient()
  const nameId = useId()
  const { theme, toggle } = useTheme()
  // 殻の下では, アプリの名前と明暗の切り替えを殻の上の帯が持つ
  const { embedded } = useStage()
  const [name, setName] = useState(initialName ?? fileName)
  const [dialog, setDialog] = useState<'open' | 'yaml' | 'python' | 'project' | null>(null)

  // 保存した時点の定義を, 保存していない変更の判定に使う
  const save = useMutation({
    mutationFn: (target: Spec & { name: string }) => api.saveModel(target.name, target),
    onSuccess: (summary, target) => {
      void queryClient.invalidateQueries({ queryKey: ['models'] })
      onSaved(summary.name, target)
    },
  })
  // 開いたものと違う名前で保存する場合, 同じ名前の保存した定義があれば
  // 上書きしてよいかを確かめる (一覧を読めない場合は確かめずに保存)
  // 確かめている間は保存のボタンを止め, 押した時点の名前で保存する
  const [asking, setAsking] = useState(false)
  const requestSave = async () => {
    const target = { ...whole, name }
    if (name !== ownFile) {
      setAsking(true)
      const models = await queryClient
        .fetchQuery({ queryKey: ['models'], queryFn: api.listModels, staleTime: 0 })
        .catch(() => [])
        .finally(() => setAsking(false))
      if (
        models.some((model) => model.name === target.name) &&
        !window.confirm(`保存した定義 ${target.name} があります. 上書きしますか?`)
      ) {
        return
      }
    }
    save.mutate(target)
  }
  // 保存している間を外に知らせる (作り直す前に止まった場合も戻す)
  const pending = save.isPending || asking
  useEffect(() => {
    onSavingChange?.(pending)
    return () => onSavingChange?.(false)
  }, [pending, onSavingChange])
  const open = useMutation({
    mutationFn: ({
      source,
      target,
      project,
      root,
    }: {
      source: 'model' | 'example' | 'experiment'
      target: string
      /** 実験の写しのプロジェクトと置き場 */
      project?: string
      root?: string
    }) =>
      source === 'model'
        ? api.getModel(target)
        : source === 'experiment'
          ? api.openExperimentSpec(project ?? '', target, root ?? '')
          : api.getExample(target),
    onSuccess: (model, { source }) => {
      onLoad(model.name, model.spec as unknown as Spec, source)
      setDialog(null)
    },
  })
  const exported = useMutation({
    mutationFn: () => api.exportPython(whole, 'Model'),
  })

  // YAML は今の定義のもの. 部分モデルを編集している間の確認の結果は, 入力の
  // 形を本体で使う所の形に置き換えているので, 開いた時だけ書いたままの形で
  // 確かめ直す (形は推定しない)
  const subgraphYaml = useQuery({
    queryKey: ['yaml', spec, editing],
    queryFn: () => api.check(editing ? subgraphCheckTarget(spec, editing) : spec, false),
    enabled: dialog === 'yaml' && editing !== null,
  })
  const yamlText = editing ? subgraphYaml.data?.yaml : check?.yaml

  // 形を推定しなかった理由 (パラメータ数の上限か, 入力の例が大きすぎる場合)
  const notInferred = check?.estimated_only
    ? 'パラメータ数が上限を超えるので, 形は推定していません'
    : check?.issues.find((issue) => issue.code === 'input_too_large')?.message

  // 誤りは, 定義の誤りと形の推定で失敗したノードの数
  const errors =
    (check?.issues ?? []).filter((issue) => issue.severity === 'error').length +
    failedNodes(check, spec).length
  const validName = /^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$/.test(name)

  // 入力と出力は, 使われていない名前で右下に追加
  // 入力は一番左のノードの左, 出力は一番右のノードの右 (流れの向きに合わせる)
  const placed = Object.values(spec.nodes).flatMap((node) =>
    node.ui?.x !== undefined && node.ui.y !== undefined ? [{ x: node.ui.x, y: node.ui.y }] : [],
  )
  const leftOf = () => {
    if (placed.length === 0) return undefined
    const first = placed.reduce((a, b) => (b.x < a.x ? b : a))
    return { x: first.x - 260, y: first.y }
  }
  const rightOf = () => {
    if (placed.length === 0) return undefined
    const last = placed.reduce((a, b) => (b.x > a.x ? b : a))
    return { x: last.x + 280, y: last.y }
  }

  const addInput = () => {
    const input = freshName(spec, 'input')
    onChange({
      ...spec,
      inputs: { ...spec.inputs, [input]: { shape: [4, 16], ui: center(undefined, leftOf()) } },
    })
    onSelect(input)
  }
  const addOutput = () => {
    const used = new Set([...Object.keys(spec.outputs), ...Object.keys(spec.ui?.outputs ?? {})])
    let index = 1
    while (used.has(`out${index}`)) index += 1
    const output = `out${index}`
    // どこにもつながっていないノードが 1 つだけで, 出力の口が 1 つなら
    // それにつなぐ (出力の口が複数の部品は口を選ぶ必要がある)
    const dangling = danglingNodes(spec).filter(
      (name) => (units.get(spec.nodes[name]?.unit ?? '')?.outputs.length ?? 0) === 1,
    )
    onChange({
      ...spec,
      outputs: dangling.length === 1 ? { ...spec.outputs, [output]: dangling[0]! } : spec.outputs,
      ui: {
        ...spec.ui,
        outputs: { ...(spec.ui?.outputs ?? {}), [output]: center(undefined, rightOf()) },
      },
    })
    onSelect(`output:${output}`)
  }

  return (
    <header className="flex flex-wrap items-center gap-2 border-b border-line bg-surface px-3 py-2">
      {/* 狭い画面では名前を隠し, 上の操作を 1 行に収める (読み上げには残す) */}
      {!embedded && (
        <h1 className="mr-2 text-sm font-semibold max-xl:sr-only">ML-CreateArchitecture</h1>
      )}
      {/* 保存の結果が別の定義に反映されないよう, 保存の間は新規と開くを止める */}
      <Button
        variant="secondary"
        icon={FileText}
        disabled={save.isPending}
        onClick={() => {
          if (!dirty || window.confirm('保存していない変更を捨てて, 新しい定義を作りますか?')) {
            onLoad('model', emptySpec(), 'new')
          }
        }}
        aria-label="新規"
        title="新規"
      >
        <span className="max-xl:sr-only">新規</span>
      </Button>
      <Button
        variant="secondary"
        icon={FolderOpen}
        disabled={save.isPending}
        onClick={() => setDialog('open')}
        data-return="open"
        aria-label="開く"
        title="開く"
      >
        <span className="max-xl:sr-only">開く</span>
      </Button>
      <label htmlFor={nameId} className="sr-only">
        保存する名前
      </label>
      <input
        id={nameId}
        value={name}
        onChange={(event) => {
          setName(event.target.value)
          onNameChange?.(event.target.value)
        }}
        aria-invalid={!validName}
        aria-describedby={validName ? undefined : `${nameId}-error`}
        className="h-8 w-40 rounded-md border border-line bg-bg px-2 font-mono text-xs max-xl:w-28 aria-[invalid=true]:border-critical"
      />
      {!validName && (
        <span id={`${nameId}-error`} role="alert" className="text-[11px] text-critical-text">
          名前は英数字, _, - (64 文字まで)
        </span>
      )}
      <Button
        icon={Save}
        data-save
        disabled={!validName || save.isPending || asking}
        onClick={() => void requestSave()}
        title={validName ? undefined : '名前は英数字, _, - にしてください'}
      >
        保存{dirty ? ' *' : ''}
      </Button>
      {/* 保存の知らせは, 上の操作をずらさないように画面の右下に出す */}
      <span
        role="status"
        className="fixed right-4 bottom-4 z-50 rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-md empty:hidden"
      >
        {savedNotice}
      </span>
      <span className="mx-1 h-6 w-px bg-line" aria-hidden />
      <Button
        variant="ghost"
        icon={Undo2}
        disabled={!onUndo}
        onClick={onUndo}
        aria-label="元に戻す"
        title="元に戻す (Ctrl+Z, Mac は Cmd+Z)"
        aria-keyshortcuts="Control+Z Meta+Z"
      />
      <Button
        variant="ghost"
        icon={Redo2}
        disabled={!onRedo}
        onClick={onRedo}
        aria-label="やり直す"
        title="やり直す (Ctrl+Shift+Z, Mac は Cmd+Shift+Z)"
        aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z Control+Y"
      />
      <Button
        variant="ghost"
        icon={Network}
        onClick={onArrange}
        title="つながりの順に左から右へ並べ直す (元に戻すで戻せる)"
      >
        <span className="max-2xl:sr-only">整列</span>
      </Button>
      <span className="mx-1 h-6 w-px bg-line" aria-hidden />
      <Button variant="secondary" icon={LogIn} onClick={addInput} title="入力を加える">
        <span className="max-2xl:sr-only">入力</span>
      </Button>
      <Button variant="secondary" icon={LogOut} onClick={addOutput} title="出力を加える">
        <span className="max-2xl:sr-only">出力</span>
      </Button>
      <span className="mx-1 h-6 w-px bg-line" aria-hidden />
      <Button
        variant="ghost"
        icon={FileText}
        onClick={() => setDialog('yaml')}
        data-return="yaml"
        title="保存する形 (YAML) を見る"
      >
        <span className="max-2xl:sr-only">YAML</span>
      </Button>
      <Button
        variant="ghost"
        icon={Code2}
        onClick={() => {
          exported.mutate()
          setDialog('python')
        }}
        data-return="python"
        title="同じ計算をする Python のコードを見る"
      >
        <span className="max-2xl:sr-only">Python</span>
      </Button>
      {/* 書き出しの操作に並べる. 上の操作を 1 行に収めるため, 文字は幅 1800 px 以上だけ */}
      <Button
        variant="ghost"
        icon={FolderOutput}
        onClick={() => setDialog('project')}
        data-return="project"
        aria-label="データセットとプロジェクト"
        title="データセットから入力と出力を作る, 学習のプロジェクトに保存する"
      >
        <span className="max-[1800px]:sr-only">データセットとプロジェクト</span>
      </Button>
      <div className="ml-auto flex items-center gap-3 text-xs" aria-live="polite">
        {checking && <span className="text-ink-3">確認中…</span>}
        {/* 大きすぎて形を推定していない場合は, 誤りが無くても知らせる */}
        {!checking && notInferred && (
          <span className="flex items-center gap-1 text-warning-text" title={notInferred}>
            <AlertTriangle className="size-3.5" aria-hidden />
            形は未推定
          </span>
        )}
        <span>
          <span className="max-xl:sr-only">
            {editing ? `パラメータ数 (sub.${editing})` : 'パラメータ数'}{' '}
          </span>
          <span
            className="font-mono"
            title={check?.params != null ? `${check.params.toLocaleString()} 個` : undefined}
          >
            {formatCount(check?.params)}
          </span>
        </span>
        {/* 部品がまだ無い (新規の直後) は, 誤りの数ではなく始め方を出す */}
        {Object.keys(spec.nodes).length === 0 ? (
          <span className="text-ink-3">部品なし</span>
        ) : (
          <span
            className={errors > 0 ? 'flex items-center gap-1 text-critical-text' : 'text-ink-3'}
          >
            {errors > 0 && <AlertTriangle className="size-3.5" aria-hidden />}
            誤り {errors} 件
          </span>
        )}
        {!embedded && (
          <Button
            variant="ghost"
            icon={theme === 'dark' ? Sun : Moon}
            onClick={toggle}
            aria-label="暗い表示"
            aria-pressed={theme === 'dark'}
            title={theme === 'dark' ? '明るい表示にする' : '暗い表示にする'}
          />
        )}
      </div>
      {save.error && (
        <div className="w-full">
          <ErrorBox error={save.error} />
        </div>
      )}
      {dialog === 'open' && (
        <OpenDialog
          onDeleted={(deleted) => {
            if (deleted === ownFile) onDeletedOwn()
          }}
          onOpen={(source, target) => {
            if (!dirty || window.confirm('保存していない変更を捨てて, 定義を開きますか?')) {
              open.mutate({ source, target })
            }
          }}
          onOpenExperiment={(project, run, root) => {
            if (!dirty || window.confirm('保存していない変更を捨てて, 実験の定義を開きますか?')) {
              open.mutate({ source: 'experiment', target: run, project, root })
            }
          }}
          opening={open.isPending}
          // 開けなかった誤りは, 開こうとした欄の中に出す
          experimentError={open.variables?.source === 'experiment' ? open.error : null}
          onClose={() => setDialog(null)}
          openError={open.variables?.source === 'experiment' ? null : open.error}
        />
      )}
      {dialog === 'yaml' && (
        <Dialog
          title={editing ? `YAML (部分モデル sub.${editing} の中身)` : 'YAML (保存する形)'}
          onClose={() => setDialog(null)}
        >
          <TextView
            text={yamlText || '定義を読めないので表示できません'}
            fileName={yamlText ? `${editing ? editing : name || 'model'}.yaml` : undefined}
          />
        </Dialog>
      )}
      {dialog === 'python' && (
        <Dialog title="Python のコード" onClose={() => setDialog(null)}>
          {exported.isPending ? (
            <Spinner />
          ) : exported.error ? (
            <ErrorBox error={exported.error} />
          ) : (
            <>
              {failedNodes(check, spec).length > 0 && (
                <p className="mb-2 rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning-text">
                  定義の入力の形では形の推定に失敗したノードか出力があるので,
                  このコードはそのままでは動かない可能性が高いです
                  (何も選んでいないときの詳細の一覧で理由を確かめてください).
                </p>
              )}
              <TextView
                text={exported.data?.code ?? ''}
                fileName={`${(name || 'model').replace(/-/g, '_')}.py`}
              />
            </>
          )}
        </Dialog>
      )}
      {dialog === 'project' && (
        <ProjectDialog
          spec={{ ...whole, name }}
          editing={editing}
          fresh={startsFresh(whole, dirty)}
          dirty={dirty}
          onApply={onDataset}
          onSaved={onProjectSaved}
          onTemplate={onTemplate}
          onClose={() => setDialog(null)}
        />
      )}
      {open.error && dialog !== 'open' && <ErrorBox error={open.error} />}
    </header>
  )
}
