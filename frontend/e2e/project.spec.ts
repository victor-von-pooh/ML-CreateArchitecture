import { readFileSync, rmSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { expect, test, type Page } from '@playwright/test'

// e2e/serve.sh が ML-EDA と ML-Quest で作った, 書き出したデータセット (3 列の
// 3 クラスの分類) と学習のプロジェクトの骨組み
const dataset = fileURLToPath(new URL('.work/dataset', import.meta.url))
// 灰色の画像と背景を含む 3 クラスの領域のファイルの組 (版 3. 高さと幅は決まらない)
const masks = fileURLToPath(new URL('.work/masks', import.meta.url))
const project = fileURLToPath(new URL('.work/project', import.meta.url))
const modelFile = `${project}/models/model.yaml`
// 受け持ちに合う雛形 (two_tower_retrieval) のある問題の設定のプロジェクト
const rankProject = fileURLToPath(new URL('.work/rank_project', import.meta.url))
// 異常の検知 (自己符号化器) の問題の設定のプロジェクト (記録の場所に, 学習に使う 3 列と
// 異常の行にだけ印のある正解の列の, 異常の検知の表がある)
const anomalyProject = fileURLToPath(new URL('.work/anomaly_project', import.meta.url))

// プロジェクトへの保存は 2 回目に上書きを確かめるので, 書いた順に実行
test.describe.configure({ mode: 'serial' })

// このファイルのテストがプロジェクトに書くモデルの定義
const written = ['model.yaml', 'e2e_candidate.yaml', 'e2e_picked.yaml', 'e2e_from_run.yaml'].map(
  (name) => `${project}/models/${name}`,
)

// 書く前の形から始める (直列のテストの再試行は最初のテストからやり直すが,
// 前の試行が書いたファイルは残り, 「まだ無い」の確かめが別の誤りで落ちるため)
test.beforeAll(() => {
  for (const file of written) rmSync(file, { force: true })
})

/** データセットとプロジェクトのダイアログを開く */
async function openDialog(page: Page) {
  await page.getByRole('button', { name: 'データセットとプロジェクト' }).click()
  return page.getByRole('dialog', { name: 'データセットとプロジェクト' })
}

test('データセットから入力と出力を作ると, 誤り無しで形が出る', async ({ page }) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const section = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await section.getByLabel('データセットのディレクトリ').fill(dataset)
  await section.getByRole('button', { name: '入力と出力を作る' }).click()

  // 入れた結果 (入力の形, 出す値の数, 置いた head) をダイアログに出す
  const result = section.getByRole('status')
  await expect(result).toContainText('入力 x [4, 3]')
  await expect(result).toContainText('クラス 3, 出す値 3')
  await expect(result).toContainText('全結合の head (出す値 3) を置き')
  await dialog.getByRole('button', { name: '閉じる' }).click()

  // 部品の無い定義に head が置かれ, 形の推定で [4, 3] が出る
  await expect(page.locator('[data-id="head"]')).toContainText('[4, 3]', { timeout: 20_000 })
  await expect(page.locator('header')).toContainText('誤り 0 件')

  // 元に戻すで, 入れる前 (部品の無い定義) に戻る
  await page.getByRole('button', { name: '元に戻す' }).click()
  await expect(page.locator('[data-id="head"]')).toHaveCount(0)
})

test('同梱の U-Net の例を, ファイルの組のデータセットのチャンネルとクラスの数に合わせる', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く', exact: true }).click()
  await page.getByRole('button', { name: 'unet_segmentation', exact: true }).click()
  await expect(page.locator('[data-id="head"]')).toContainText('[4, 2, 32, 32]', {
    timeout: 20_000,
  })

  const dialog = await openDialog(page)
  const section = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await section.getByLabel('データセットのディレクトリ').fill(masks)
  await section.getByRole('button', { name: '入力と出力を作る' }).click()

  // チャンネルは 1, 決まらない高さと幅は例の 32 のまま, クラスは背景を含む 3
  const result = section.getByRole('status')
  await expect(result).toContainText('入力 image [4, 1, ?, ?]')
  await expect(result).toContainText('正解 mask (mask), クラス 3, 出す値 3')
  await expect(result).toContainText('enc1a の in_channels を 1 にしました')
  await expect(result).toContainText('head の out_channels を 3 にしました')
  await dialog.getByRole('button', { name: '閉じる' }).click()
  await expect(page.locator('[data-id="head"]')).toContainText('[4, 3, 32, 32]', {
    timeout: 20_000,
  })
  await expect(page.locator('header')).toContainText('誤り 0 件')
})

test('同梱の拡散の例に表のデータセットを入れると, 残す時刻の入力のバッチをそろえて形が出る', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く', exact: true }).click()
  await page.getByRole('button', { name: 'diffusion_denoiser', exact: true }).click()
  await expect(page.locator('[data-id="head"]')).toContainText('[8, 16]', { timeout: 20_000 })

  const dialog = await openDialog(page)
  const section = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await section.getByLabel('データセットのディレクトリ').fill(dataset)
  await section.getByRole('button', { name: '入力と出力を作る' }).click()

  // データセットに無い時刻の入力 t は残し, バッチの次元を x と同じ例の大きさ 4 にする
  const result = section.getByRole('status')
  await expect(result).toContainText('入力 x [4, 3]')
  await expect(result).toContainText(
    '入力 t はデータセットに無いですが, つないでいるので残し, 最初の次元 (バッチ) を例の大きさ 4 にしました ([4])',
  )
  await dialog.getByRole('button', { name: '閉じる' }).click()

  // x_fc と t_act の和 (add) が通り, 出力 noise のつなぎ元 head が [4, 3] になる
  await expect(page.locator('[data-id="head"]')).toContainText('[4, 3]', { timeout: 20_000 })
  await expect(page.locator('header')).toContainText('誤り 0 件')
})

test('データセットでない場所は, 理由を出して定義を変えない', async ({ page }) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const section = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await section.getByLabel('データセットのディレクトリ').fill(project)
  await section.getByRole('button', { name: '入力と出力を作る' }).click()
  await expect(section.getByRole('alert')).toContainText('dataset.yaml がありません')
})

test('プロジェクトに保存すると, 学習のスクリプトが読む場所に書き, 2 回目は上書きを確かめる', async ({
  page,
}) => {
  await page.goto('/')
  let dialog = await openDialog(page)
  const data = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await data.getByLabel('データセットのディレクトリ').fill(dataset)
  await data.getByRole('button', { name: '入力と出力を作る' }).click()
  await expect(data.getByRole('status')).toContainText('出す値 3')

  // 保存先を確かめる (骨組みの config の model.spec の場所)
  let save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  await save.getByLabel('プロジェクトか問題の設定').fill(project)
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  let target = save.getByRole('group', { name: '保存先' })
  await expect(target).toContainText('models/model.yaml')
  await expect(target).not.toContainText('既にファイルがあります')

  // 保存すると知らせ, ファイルに head の出す値が書かれる
  await target.getByRole('button', { name: 'この場所に保存' }).click()
  await expect(save.getByRole('status')).toContainText('models/model.yaml に保存しました')
  expect(readFileSync(modelFile, 'utf-8')).toContain('out_features: 3')
  await dialog.getByRole('button', { name: '閉じる' }).click()

  // 開き直すと前に確かめたプロジェクトの場所を覚えていて, 既にあると出す.
  // 上書きの確認に答えてから書く
  dialog = await openDialog(page)
  save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  await expect(save.getByLabel('プロジェクトか問題の設定')).toHaveValue(project)
  await expect(
    dialog
      .getByRole('region', { name: 'データセットから入力と出力を作る' })
      .getByLabel('データセットのディレクトリ'),
  ).toHaveValue(dataset)
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  target = save.getByRole('group', { name: '保存先' })
  await expect(target).toContainText('既にファイルがあります')
  let asked = ''
  page.once('dialog', (confirm) => {
    asked = confirm.message()
    void confirm.accept()
  })
  await target.getByRole('button', { name: 'この場所に保存' }).click()
  await expect(save.getByRole('status')).toContainText('に保存しました')
  expect(asked).toContain('上書きしますか')
})

test('名前を変えてプロジェクトに保存すると, models/ の下に候補を並べ, config と既定の定義は変えない', async ({
  page,
}) => {
  const config = `${project}/configs/default/train.yaml`
  const configBefore = readFileSync(config, 'utf-8')
  const modelBefore = readFileSync(modelFile, 'utf-8')
  await page.goto('/')
  const dialog = await openDialog(page)
  const save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  await save.getByLabel('プロジェクトか問題の設定').fill(project)
  await save.getByLabel('models/ の下の名前').fill('e2e_candidate')
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  const target = save.getByRole('group', { name: '保存先' })
  await expect(target).toContainText(`${project}/models/e2e_candidate.yaml`)
  await expect(target).toContainText('モデルの定義の候補')
  await expect(target).not.toContainText('既にファイルがあります')
  await target.getByRole('button', { name: 'この場所に保存' }).click()
  await expect(save.getByRole('status')).toContainText('models/e2e_candidate.yaml に保存しました')
  expect(readFileSync(`${project}/models/e2e_candidate.yaml`, 'utf-8')).toContain('nodes')
  expect(readFileSync(modelFile, 'utf-8')).toBe(modelBefore)
  expect(readFileSync(config, 'utf-8')).toBe(configBefore)

  // 確かめた後に名前を書き換えると, 確かめ直すように出す
  await save.getByLabel('models/ の下の名前').fill('-bad')
  await expect(save).toContainText('保存先を確かめ直してください')

  // 確かめに失敗すると誤りだけを出し, 前の保存先と保存の知らせを消す
  // (先頭の - は使えない)
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  await expect(save.getByRole('alert')).toContainText('先頭に - は使えない')
  await expect(save).not.toContainText('保存先を確かめ直してください')
  await expect(save.getByRole('group', { name: '保存先' })).toHaveCount(0)
  await expect(save.getByRole('status')).not.toContainText('に保存しました')

  // 失敗した後に書き換えても, 確かめ直しの知らせは出さない (誤りのまま)
  await save.getByLabel('models/ の下の名前').fill('model')
  await expect(save.getByRole('alert')).toBeVisible()
  await expect(save).not.toContainText('保存先を確かめ直してください')

  // 既定の保存先と同じ名前は, 既定の定義を上書きすると添える
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  await expect(save.getByRole('group', { name: '保存先' })).toContainText(
    '既定の定義を上書きします',
  )

  // 名前を消すと, 既定の保存先に戻る
  await save.getByLabel('models/ の下の名前').fill('')
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  await expect(save.getByRole('group', { name: '保存先' })).toContainText('models/model.yaml')

  // 規則に合わない名前は, 理由を出して確かめない
  await save.getByLabel('models/ の下の名前').fill('../outside')
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  await expect(save.getByRole('alert')).toContainText('使えません')

  // 保存先を決められない場所 (学習のプロジェクトでないディレクトリ) も,
  // 誤りだけを出す
  await save.getByLabel('models/ の下の名前').fill('')
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  await expect(save.getByRole('group', { name: '保存先' })).toBeVisible()
  await save.getByLabel('プロジェクトか問題の設定').fill(dataset)
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  await expect(save.getByRole('alert')).toContainText('学習のプロジェクト')
  await expect(save).not.toContainText('保存先を確かめ直してください')
  await expect(save.getByRole('group', { name: '保存先' })).toHaveCount(0)
})

test('記録の無いプロジェクトでも, 保存先を確かめると config のデータセットの場所が入る', async ({
  page,
}) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  await save.getByLabel('プロジェクトか問題の設定').fill(project)
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  await expect(save.getByRole('group', { name: '保存先' })).toContainText('models/model.yaml')

  // 打つ前のデータセットの欄に, config の data.dataset (プロジェクトから) が入る
  const field = dialog
    .getByRole('region', { name: 'データセットから入力と出力を作る' })
    .getByLabel('データセットのディレクトリ')
  await expect(field).toHaveValue(`${project}/outputs/dataset`)
})

test('問題の設定を一覧から選ぶと, 記録のデータセットと保存先が入り, そのまま保存できる', async ({
  page,
}) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const list = dialog.getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })

  // プロジェクトを書き出していない問題の設定は, 理由を付けて選べない
  const draft = list.getByRole('button', { name: /e2e の下書き/ })
  await expect(draft).toBeDisabled()
  await expect(draft).toContainText('プロジェクトを書き出していません')

  // 書き出してあり, 定義の置き場の記録だけが無いものは, その理由で選べない
  const noSpec = list.getByRole('button', { name: /e2e の定義の記録なし/ })
  await expect(noSpec).toBeDisabled()
  await expect(noSpec).toContainText('書き出してありますが')

  // 打った後のデータセットの欄も, 選んだ記録の場所に変わる
  const data = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  const field = data.getByLabel('データセットのディレクトリ')
  await field.fill('/somewhere/else')
  const recorded = list.getByRole('button', { name: /e2e の売上/ })
  await recorded.click()
  await expect(recorded).toHaveAttribute('aria-pressed', 'true')

  // 保存先は記録の場所 (場所を打たずに決まる)
  const save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  await expect(save.getByLabel('プロジェクトか問題の設定')).toHaveValue(
    /problems\/recorded\/problem\.yaml$/,
  )
  const target = save.getByRole('group', { name: '保存先' })
  await expect(target).toContainText(modelFile)
  await expect(target).toContainText('問題の設定 e2e の売上 の記録の保存先')
  await expect(field).toHaveValue(dataset)

  // そのまま入力と出力を作り, 保存できる (既にあれば上書きを確かめる)
  await data.getByRole('button', { name: '入力と出力を作る' }).click()
  await expect(data.getByRole('status')).toContainText('出す値 3')
  page.on('dialog', (confirm) => void confirm.accept())
  await target.getByRole('button', { name: 'この場所に保存' }).click()
  await expect(save.getByRole('status')).toContainText(`${modelFile} に保存しました`)
  expect(readFileSync(modelFile, 'utf-8')).toContain('out_features: 3')
})

test('雛形のある問題の設定を新しく始めるときに選ぶと雛形を開き, 定義があるときは選べるようにする', async ({
  page,
}) => {
  await page.goto('/')
  let dialog = await openDialog(page)
  let rank = dialog
    .getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
    .getByRole('button', { name: /e2e の推薦/ })
  await expect(rank).toContainText('雛形: two_tower_retrieval')

  // 新規の画面で選ぶと, 雛形を開いたまま続けられる (保存先は今までどおり記録の場所)
  await rank.click()
  let note = dialog.getByRole('region', { name: '受け持ちに合う雛形' })
  await expect(note.getByRole('status')).toContainText(
    '問題の設定 e2e の推薦 の受け持ちに合う雛形 two_tower_retrieval を開きました',
  )
  await expect(note.getByRole('button')).toHaveCount(0)
  const save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  await expect(save.getByRole('group', { name: '保存先' })).toContainText(
    `${rankProject}/models/model.yaml`,
  )
  await dialog.getByRole('button', { name: '閉じる' }).click()
  await expect(page.locator('[data-id="user_fc1"]')).toBeVisible({ timeout: 20_000 })
  // まだどこにも保存していないので, 保存していない変更になる (知らせはダイアログの中だけ)
  await expect(page.getByRole('button', { name: '保存 *' })).toBeVisible()
  await expect(page.getByText('雛形 two_tower_retrieval を開きました')).toHaveCount(0)

  // 既に定義がある場合は上書きせず, 「雛形を開く」で選ぶ
  await page.goto('/')
  await page.getByRole('button', { name: '開く', exact: true }).click()
  await page.getByRole('button', { name: 'mlp', exact: true }).click()
  await expect(page.locator('[data-id="fc1"]')).toBeVisible({ timeout: 20_000 })
  dialog = await openDialog(page)
  rank = dialog
    .getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
    .getByRole('button', { name: /e2e の推薦/ })
  await rank.click()
  note = dialog.getByRole('region', { name: '受け持ちに合う雛形' })
  await expect(note.getByRole('status')).toContainText('今の定義があるので開いていません')
  await expect(page.locator('[data-id="fc1"]')).toHaveCount(1)
  // 保存していない変更が無くても, 今の定義を置き換えるので確かめる
  let asked = ''
  page.once('dialog', (confirm) => {
    asked = confirm.message()
    void confirm.accept()
  })
  await note.getByRole('button', { name: '雛形 two_tower_retrieval を開く' }).click()
  await expect(note.getByRole('status')).toContainText('を開きました')
  expect(asked).toContain('今の定義を雛形で置き換えますか')
  await dialog.getByRole('button', { name: '閉じる' }).click()
  await expect(page.locator('[data-id="user_fc1"]')).toBeVisible({ timeout: 20_000 })
  await expect(page.locator('[data-id="fc1"]')).toHaveCount(0)

  // 開いた雛形は保存していないので, 閉じるときに確かめる
  const unload = new Promise<string>((resolve) =>
    page.once('dialog', (confirm) => {
      resolve(confirm.type())
      void confirm.accept()
    }),
  )
  await page.close({ runBeforeUnload: true })
  expect(await unload).toBe('beforeunload')
})

test('異常の検知の問題の設定を新しく始めるときに選ぶと自己符号化器の雛形を開き, 入力と出力の幅を学習に使う列の数にする', async ({
  page,
}) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const problems = dialog.getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
  const anomaly = problems.getByRole('button', { name: /e2e の異常の検知/ })
  await expect(anomaly).toContainText('雛形: tabular_autoencoder')

  // 距離に基づく度合いは雛形の名前を出さず, 定義が要らない理由で選べない
  const distance = problems.getByRole('listitem').filter({ hasText: 'e2e の距離の度合い' })
  await expect(distance).toContainText('モデルの定義は要りません')
  await expect(distance).not.toContainText('雛形:')

  // 新規の画面で選ぶと雛形を開き, 保存先は記録の場所
  await anomaly.click()
  const note = dialog.getByRole('region', { name: '受け持ちに合う雛形' })
  await expect(note.getByRole('status')).toContainText(
    '問題の設定 e2e の異常の検知 の受け持ちに合う雛形 tabular_autoencoder を開きました',
  )
  await expect(note.getByRole('button')).toHaveCount(0)
  await expect(
    dialog
      .getByRole('region', { name: 'プロジェクトに保存' })
      .getByRole('group', { name: '保存先' }),
  ).toContainText(`${anomalyProject}/models/model.yaml`)

  // 記録のデータセットから入力と出力を作ると, 幅を学習に使う 3 列にする (正解の列は数えない)
  const data = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await expect(data.getByLabel('データセットのディレクトリ')).toHaveValue(
    `${anomalyProject}/outputs/dataset`,
  )
  await data.getByRole('button', { name: '入力と出力を作る' }).click()
  const result = data.getByRole('status')
  await expect(result).toContainText(
    'データセットから入れました: 入力 x [4, 3], タスク anomaly, 目的の列 is_anomaly, 出す値 3',
  )
  await expect(result).toContainText(
    '問題の設定 e2e の異常の検知 が異常の検知 (自己符号化器)なので, 入力と出力の幅を, 学習に使う列の数 3 にしました ' +
      '(自己符号化器は入力を作り直すので, 出力を入力と同じ幅にします. 正解の列 is_anomaly は数えません)',
  )
  await expect(result).toContainText('enc1 の in_features を 3 にしました')
  await expect(result).toContainText('dec2 の out_features を 3 にしました')
  await dialog.getByRole('button', { name: '閉じる' }).click()

  // 縮めた幅は変えず, 再構成は入力と同じ形 (誤り 0 件は確認の結果が届く前にも出るので,
  // 入れた後の形を先に待つ)
  await expect(page.locator('[data-id="dec2"]')).toContainText('[4, 3]', { timeout: 20_000 })
  await expect(page.locator('[data-id="enc2"]')).toContainText('[4, 4]')
  await expect(page.locator('header')).toContainText('誤り 0 件')
  await expect(page.getByRole('button', { name: '保存 *' })).toBeVisible()
})

test('並べるの問題の設定を選んで入力と出力を作ると, 入力の大きさを学習用の利用者と品物の数にし, 決め方を出す', async ({
  page,
}) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  await dialog
    .getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
    .getByRole('button', { name: /e2e の推薦/ })
    .click()
  await expect(
    dialog.getByRole('region', { name: '受け持ちに合う雛形' }).getByRole('status'),
  ).toContainText('雛形 two_tower_retrieval を開きました')

  // 欄には記録の反応の表のファイルが入り, 問題の設定も渡す
  const data = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await expect(data.getByLabel('データセットのディレクトリ')).toHaveValue(
    /rank_project\/outputs\/interactions\.csv$/,
  )
  await expect(data).toContainText('並べるなら, 入力 user と item の大きさを')
  await data.getByRole('button', { name: '入力と出力を作る' }).click()

  // 反応の表の 7 人と 5 品のうち, 学習用にいる 6 人と 4 品にする (決め方を初めに出す)
  const result = data.getByRole('status')
  await expect(result).toContainText('反応の表から入れました: 入力 user [4, 6], item [4, 4]')
  await expect(result).toContainText(
    '問題の設定 e2e の推薦 が並べるなので, 入力 user の大きさを学習用の利用者の数 6, ' +
      'item の大きさを学習用の品物の数 4 にしました',
  )
  await expect(result).toContainText(
    '(学習用の反応の行の数 32 (時刻の順に分けたもの). 分ける割合はプロジェクトの学習の設定の検証用 0.1 とテスト用 0.1',
  )
  await expect(result).toContainText('user_fc1 の in_features を 6 にしました')
  await expect(result).toContainText('item_fc1 の in_features を 4 にしました')
  await dialog.getByRole('button', { name: '閉じる' }).click()
  await expect(page.locator('[data-id="user_fc1"]')).toBeVisible({ timeout: 20_000 })
  await expect(page.locator('header')).toContainText('誤り 0 件', { timeout: 20_000 })
})

test('名前を書いてから問題の設定を一覧から選ぶと, 記録のプロジェクトの models/ の下の名前が保存先になる', async ({
  page,
}) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  await save.getByLabel('models/ の下の名前').fill('e2e_picked')
  await dialog
    .getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
    .getByRole('button', { name: /e2e の売上/ })
    .click()
  const target = save.getByRole('group', { name: '保存先' })
  await expect(target).toContainText(`${project}/models/e2e_picked.yaml`)
  await expect(target).toContainText('モデルの定義の候補')
})

test('確かめが重なった場合は最後の確かめの結果だけを出し, 確かめに失敗すると一覧の選択を外す', async ({
  page,
}) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const list = dialog.getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
  const recorded = list.getByRole('button', { name: /e2e の売上/ })
  const save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  const field = save.getByLabel('プロジェクトか問題の設定')
  const button = save.getByRole('button', { name: /保存先を確かめる|確かめています/ })
  const target = save.getByRole('group', { name: '保存先' })

  // 次の 1 回の確かめを遅らせる (late が成功か誤りで返す). 返し終わるまで待つ
  // ための Promise を包んで返す (async の関数から Promise をそのまま返すと,
  // 呼ぶ側の await が返し終わるまで待つため)
  const delayNext = async (late: 'success' | 'error') => {
    let finish = () => {}
    const finished = new Promise<void>((resolve) => (finish = resolve))
    await page.route(
      '**/api/project/target',
      async (route) => {
        const response = late === 'success' ? await route.fetch() : null
        await new Promise((resolve) => setTimeout(resolve, 1500))
        await (response
          ? route.fulfill({ response })
          : route.fulfill({ status: 422, json: { detail: '遅れて返った誤り' } }))
        finish()
      },
      { times: 1 },
    )
    return { finished }
  }

  // 打った場所の確かめが遅れて成功で返っても, 後に一覧から選んだ確かめの
  // 結果のまま (確かめ直しの知らせを出さず, 一覧の選択も外さない)
  let delayed = await delayNext('success')
  await field.fill(project)
  await button.click()
  await expect(button).toHaveText('確かめています…')
  await recorded.click()
  await delayed.finished
  await page.waitForTimeout(300)
  await expect(field).toHaveValue(/problems\/recorded\/problem\.yaml$/)
  await expect(target).toContainText('問題の設定 e2e の売上 の記録の保存先')
  await expect(save).not.toContainText('保存先を確かめ直してください')
  await expect(recorded).toHaveAttribute('aria-pressed', 'true')

  // 遅れて誤りで返っても, 後の確かめの結果のまま
  delayed = await delayNext('error')
  await field.fill(project)
  await button.click()
  await expect(button).toHaveText('確かめています…')
  await recorded.click()
  await delayed.finished
  await page.waitForTimeout(300)
  await expect(save.getByRole('alert')).toHaveCount(0)
  await expect(target).toContainText('問題の設定 e2e の売上 の記録の保存先')
  await expect(recorded).toHaveAttribute('aria-pressed', 'true')

  // 確かめに失敗すると, 一覧の選択も外す (先頭の - は使えない)
  await save.getByLabel('models/ の下の名前').fill('-bad')
  await button.click()
  await expect(save.getByRole('alert')).toContainText('先頭に - は使えない')
  await expect(target).toHaveCount(0)
  await expect(recorded).toHaveAttribute('aria-pressed', 'false')
})

test('実験の定義の写しを開き, 名前を付けてプロジェクトの models/ に保存する', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く', exact: true }).click()
  const open = page.getByRole('dialog', { name: '定義を開く' })
  const copies = open.getByRole('region', { name: '学習の実験の定義の写し' })
  await copies.getByLabel('プロジェクトか問題の設定').fill(project)
  await copies.getByRole('button', { name: '実験を探す' }).click()
  const run = copies.getByRole('button', { name: /e2e_run\/2026-11-20\/10-00-00/ })
  await expect(run).toContainText('models/e2e_run.yaml')
  await expect(run).toContainText('完了')
  await run.click()

  // 写しの定義が開き, 保存の仕方を知らせる
  await expect(open).toHaveCount(0)
  await expect(page.locator('[data-id="e2e_fc"]')).toContainText('[4, 2]', { timeout: 20_000 })
  await expect(page.getByText('実験の定義の写しを開きました')).toBeVisible()

  // 今の保存の流れで, 名前を付けて models/ に書く (探したプロジェクトの場所が入る)
  const dialog = await openDialog(page)
  const save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  await expect(save.getByLabel('プロジェクトか問題の設定')).toHaveValue(project)
  await save.getByLabel('models/ の下の名前').fill('e2e_from_run')
  await save.getByRole('button', { name: '保存先を確かめる' }).click()
  const target = save.getByRole('group', { name: '保存先' })
  await expect(target).toContainText(`${project}/models/e2e_from_run.yaml`)
  await target.getByRole('button', { name: 'この場所に保存' }).click()
  await expect(save.getByRole('status')).toContainText('models/e2e_from_run.yaml に保存しました')
  expect(readFileSync(`${project}/models/e2e_from_run.yaml`, 'utf-8')).toContain('e2e_fc')
})
