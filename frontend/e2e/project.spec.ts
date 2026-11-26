import { existsSync, readFileSync, rmSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { expect, test, type Locator, type Page } from '@playwright/test'

// e2e/serve.sh が ML-EDA と ML-Quest で作った, 書き出したデータセット (3 列の
// 3 クラスの分類) と学習のプロジェクトの骨組み
const dataset = fileURLToPath(new URL('.work/dataset', import.meta.url))
// 灰色の画像と背景を含む 3 クラスの領域のファイルの組 (版 3. 高さと幅は決まらない)
const masks = fileURLToPath(new URL('.work/masks', import.meta.url))
// 128 点の観測の点群から 64 点の点群を当てるファイルの組 (版 2)
const clouds = fileURLToPath(new URL('.work/clouds', import.meta.url))
const project = fileURLToPath(new URL('.work/project', import.meta.url))
const modelFile = `${project}/models/model.yaml`
// 受け持ちに合う雛形 (two_tower_retrieval) のある問題の設定のプロジェクト
const rankProject = fileURLToPath(new URL('.work/rank_project', import.meta.url))
// 異常の検知 (自己符号化器) の問題の設定のプロジェクト (記録の場所に, 学習に使う 3 列と
// 異常の行にだけ印のある正解の列の, 異常の検知の表がある)
const anomalyProject = fileURLToPath(new URL('.work/anomaly_project', import.meta.url))
// 効果の見積もりの問題の設定のプロジェクト (記録の場所に, 効果の見積もりに使う 2 列と処置の列
// と結果の列の, 介入の記録の表がある)
const upliftProject = fileURLToPath(new URL('.work/uplift_project', import.meta.url))
// 値段を変える判断の問題の設定のプロジェクト (記録の場所に, 特徴の 2 列と値段の列と, 売れた数,
// 原価, 商品の id の列の, 値段を変えた記録の表がある)
const priceProject = fileURLToPath(new URL('.work/price_project', import.meta.url))
// 時間をまたぐ量の判断の問題の設定のプロジェクト (記録の場所に, 学習に使う 5 列と, 売れた数,
// 商品の id, 日付, 在庫の数の列の, 需要の記録の表がある. 学習の設定の学ぶ水準は 4 つ)
const demandProject = fileURLToPath(new URL('.work/demand_project', import.meta.url))

// プロジェクトへの保存は 2 回目に上書きを確かめるので, 書いた順に実行
test.describe.configure({ mode: 'serial' })

// このファイルのテストがプロジェクトに書くモデルの定義
const written = [
  ...['model.yaml', 'e2e_candidate.yaml', 'e2e_picked.yaml', 'e2e_from_run.yaml'].map(
    (name) => `${project}/models/${name}`,
  ),
  ...[anomalyProject, upliftProject, priceProject, demandProject, rankProject].map(
    (place) => `${place}/models/model.yaml`,
  ),
]

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

/**
 * 今の定義をプロジェクトに保存すると, 骨組みの入力と出力に合わない所 (直す場所と文) が出て,
 * 書かないことを確かめ, 合わない所の欄を返す
 */
async function expectContract(dialog: Locator, file: string, places: string[]) {
  const save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  await save
    .getByRole('group', { name: '保存先' })
    .getByRole('button', { name: 'この場所に保存' })
    .click()
  const findings = save.getByRole('region', { name: '骨組みの入力と出力に合わない所' })
  await expect(findings.getByRole('alert')).toContainText(
    `骨組みの学習のスクリプトが読む入力と出力に合わない所が ${places.length} 件あります`,
  )
  for (const place of places) await expect(findings).toContainText(place)
  await expect(findings.getByRole('listitem')).toHaveCount(places.length)
  expect(existsSync(file)).toBe(false)
  return findings
}

/**
 * データセットから作った定義をプロジェクトに保存すると, 合わない所を出さずに書く
 * (overwrite は既に書いた定義があり, 上書きを確かめる場合)
 */
async function expectSaved(page: Page, dialog: Locator, file: string, overwrite = false) {
  const save = dialog.getByRole('region', { name: 'プロジェクトに保存' })
  let asked = ''
  if (overwrite) {
    page.once('dialog', (confirm) => {
      asked = confirm.message()
      void confirm.accept()
    })
  }
  await save
    .getByRole('group', { name: '保存先' })
    .getByRole('button', { name: 'この場所に保存' })
    .click()
  await expect(save.getByRole('status')).toContainText(`${file} に保存しました`)
  await expect(save.getByRole('region', { name: '骨組みの入力と出力に合わない所' })).toHaveCount(0)
  if (overwrite) expect(asked).toContain('上書きしますか')
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
  await expect(result).toContainText(
    '定義は雛形 unet_segmentation のままなので, データに合わせる引数を ML-ModelControl の書き換え',
  )
  await expect(result).toContainText(
    '雛形の引数 channels (チャンネル数) を 3 から 1 にしました (入力 image の形の 2 番目, enc1a の in_channels)',
  )
  await expect(result).toContainText(
    '雛形の引数 classes (クラス数 (背景を含む)) を 2 から 3 にしました (head の out_channels)',
  )
  await dialog.getByRole('button', { name: '閉じる' }).click()
  await expect(page.locator('[data-id="head"]')).toContainText('[4, 3, 32, 32]', {
    timeout: 20_000,
  })
  await expect(page.locator('header')).toContainText('誤り 0 件')
})

test('同梱の点群の例を, ファイルの組のデータセットの入力の点の数と出す点の数に合わせる', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('button', { name: '開く', exact: true }).click()
  await page.getByRole('button', { name: 'point_cloud_predictor', exact: true }).click()
  await expect(page.locator('[data-id="cloud"]')).toContainText('[4, 256, 3]', {
    timeout: 20_000,
  })

  const dialog = await openDialog(page)
  const section = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await section.getByLabel('データセットのディレクトリ').fill(clouds)
  await section.getByRole('button', { name: '入力と出力を作る' }).click()

  // 入力の点の数 128 と出す点の数 64 を, 雛形の書き換えで関わる場所にそろえて入れる
  const result = section.getByRole('status')
  await expect(result).toContainText('入力 observed [4, 128, 3]')
  await expect(result).toContainText(
    '定義は雛形 point_cloud_predictor のままなので, データに合わせる引数を ML-ModelControl の書き換え',
  )
  await expect(result).toContainText(
    '雛形の引数 points_in (入力の点の数) を 512 から 128 にしました (入力 observed の形の 2 番目)',
  )
  await expect(result).toContainText(
    '雛形の引数 points_out (出す点の数) を 256 から 64 にしました (decode2 の out_features (× 3), cloud の shape の 1 番目)',
  )
  await dialog.getByRole('button', { name: '閉じる' }).click()
  await expect(page.locator('[data-id="cloud"]')).toContainText('[4, 64, 3]', {
    timeout: 20_000,
  })
  await expect(page.locator('header')).toContainText('誤り 0 件')

  // 入れた後の雛形にも, 入れ直せる (今の値を雛形の引数として読む)
  const again = await openDialog(page)
  const retry = again.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await retry.getByLabel('データセットのディレクトリ').fill(clouds)
  await retry.getByRole('button', { name: '入力と出力を作る' }).click()
  await expect(retry.getByRole('status')).toContainText(
    '雛形の引数 points_out (出す点の数) は 64 のままです',
  )
  await again.getByRole('button', { name: '閉じる' }).click()
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
  await expect(result).toContainText(
    '定義は雛形 diffusion_denoiser のままなので, データに合わせる引数を ML-ModelControl の書き換え',
  )
  await expect(result).toContainText(
    '雛形の引数 features (特徴の数) を 16 から 3 にしました (入力 x の形の 2 番目, x_fc の in_features, head の out_features)',
  )
  await expect(result).toContainText(
    'データセットの出す値の数 3 は使っていません (雛形 diffusion_denoiser の出力の大きさは雛形の決まりで決まります.',
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

  // 雛形のままの幅 16 は骨組みが読む学習に使う列の数 3 と違うので, 直す場所を出して書かない.
  // 再構成の幅は形の推定で求めて比べる. 書くかは利用者が選べる (直さずに保存)
  const anomalyFile = `${anomalyProject}/models/model.yaml`
  const findings = await expectContract(dialog, anomalyFile, [
    'inputs.x: 入力 x の最後の大きさを学習に使う列の数 3 にしてください (今は 16)',
    'outputs.recon: 出力 recon の最後の大きさを学習に使う列の数 3 にしてください (今は 16)',
  ])
  await findings.getByRole('button', { name: '直さずに保存' }).click()
  await expect(
    dialog.getByRole('region', { name: 'プロジェクトに保存' }).getByRole('status'),
  ).toContainText(`${anomalyFile} に保存しました`)
  expect(readFileSync(anomalyFile, 'utf-8')).toContain('in_features: 16')

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
  await expect(result).toContainText(
    '定義は雛形 tabular_autoencoder のままなので, データに合わせる引数を ML-ModelControl の書き換え',
  )
  await expect(result).toContainText(
    '雛形の引数 features (特徴の数) を 16 から 3 にしました (入力 x の形の 2 番目, enc1 の in_features, dec2 の out_features)',
  )

  // 作った定義は骨組みに合うので, 合わない所を出さずに書く
  await expectSaved(page, dialog, anomalyFile, true)
  expect(readFileSync(anomalyFile, 'utf-8')).toContain('in_features: 3')
  await dialog.getByRole('button', { name: '閉じる' }).click()

  // 縮めた幅は変えず, 再構成は入力と同じ形 (誤り 0 件は確認の結果が届く前にも出るので,
  // 入れた後の形を先に待つ)
  await expect(page.locator('[data-id="dec2"]')).toContainText('[4, 3]', { timeout: 20_000 })
  await expect(page.locator('[data-id="enc2"]')).toContainText('[4, 4]')
  await expect(page.locator('header')).toContainText('誤り 0 件')
  await expect(page.getByRole('button', { name: '保存 *' })).toBeVisible()
})

test('効果の見積もりの問題の設定を新しく始めるときに選ぶと効果の雛形を開き, 入力の幅を処置の列を除いた列の数にする', async ({
  page,
}) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const uplift = dialog
    .getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
    .getByRole('button', { name: /e2e のクーポンの効果/ })
  await expect(uplift).toContainText('雛形: tabular_uplift')

  // 新規の画面で選ぶと雛形を開き, 保存先は記録の場所
  await uplift.click()
  const note = dialog.getByRole('region', { name: '受け持ちに合う雛形' })
  await expect(note.getByRole('status')).toContainText(
    '問題の設定 e2e のクーポンの効果 の受け持ちに合う雛形 tabular_uplift を開きました',
  )
  await expect(note.getByRole('button')).toHaveCount(0)
  await expect(
    dialog
      .getByRole('region', { name: 'プロジェクトに保存' })
      .getByRole('group', { name: '保存先' }),
  ).toContainText(`${upliftProject}/models/model.yaml`)

  // 雛形のままの幅は骨組みが読む数と違うので, 直す場所を出して書かない
  const file = `${upliftProject}/models/model.yaml`
  await expectContract(dialog, file, [
    'inputs.x: 入力 x の最後の大きさを処置の列を除いた学習に使う列の数 2 にしてください (今は 16)',
  ])

  // 記録のデータセットから入力と出力を作ると, 入力の幅を 2 列にし (処置の列, 結果の列,
  // 行の id の列は数えない), 2 つの出力はどちらも 1
  const data = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await expect(data.getByLabel('データセットのディレクトリ')).toHaveValue(
    `${upliftProject}/outputs/dataset`,
  )
  await data.getByRole('button', { name: '入力と出力を作る' }).click()
  const result = data.getByRole('status')
  await expect(result).toContainText(
    'データセットから入れました: 入力 x [4, 2], タスク uplift, 目的の列 purchased, 出す値 1',
  )
  await expect(result).toContainText(
    '問題の設定 e2e のクーポンの効果 が効果の見積もりなので, 入力の幅を, 学習に使う列の数 2 にしました. ' +
      '入力に入れない列: 処置 (行動したかどうか. クーポンを配ったかなど) の列 coupon, ' +
      '結果 (行動の後にどうなったか. 買ったかなど) の列 purchased, 行を見分ける id の列 customer_id ' +
      '(効果は処置したときとしないときの結果の差として学ぶので, 処置と結果は入力に入れません). ' +
      '効果の見積もりの 2 つの出力 (処置したときの結果の予測と, しないときの結果の予測) は, ' +
      'どちらも 1 行に 1 つの値を出します',
  )
  await expect(result).toContainText(
    '定義は雛形 tabular_uplift のままなので, データに合わせる引数を ML-ModelControl の書き換え',
  )
  await expect(result).toContainText(
    '雛形の引数 features (特徴の数) を 16 から 2 にしました (入力 x の形の 2 番目, treated1 の in_features, control1 の in_features)',
  )
  await expect(result).toContainText('出力 treated と control の値の数は, 雛形の 1 のままです')

  // 作った定義は骨組みに合うので, 合わない所を出さずに書く
  await expectSaved(page, dialog, file)
  await dialog.getByRole('button', { name: '閉じる' }).click()

  // 隠れ層の幅は変えず, 2 つの出力はどちらも (バッチ, 1) (誤り 0 件は確認の結果が届く前にも
  // 出るので, 入れた後の形を先に待つ)
  await expect(page.locator('[data-id="treated1"]')).toContainText('[4, 32]', {
    timeout: 20_000,
  })
  await expect(page.locator('[data-id="treated2"]')).toContainText('[4, 1]')
  await expect(page.locator('[data-id="control2"]')).toContainText('[4, 1]')
  await expect(page.locator('header')).toContainText('誤り 0 件')
  await expect(page.getByRole('button', { name: '保存 *' })).toBeVisible()
})

test('値段を変える判断の問題の設定を新しく始めるときに選ぶと売れ行きの雛形を開き, 入力の幅を特徴の列と値段の列の数にする', async ({
  page,
}) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const price = dialog
    .getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
    .getByRole('button', { name: /e2e の値段の見直し/ })
  await expect(price).toContainText('雛形: tabular_price')

  // 新規の画面で選ぶと雛形を開き, 保存先は記録の場所
  await price.click()
  const note = dialog.getByRole('region', { name: '受け持ちに合う雛形' })
  await expect(note.getByRole('status')).toContainText(
    '問題の設定 e2e の値段の見直し の受け持ちに合う雛形 tabular_price を開きました',
  )
  await expect(note.getByRole('button')).toHaveCount(0)
  await expect(
    dialog
      .getByRole('region', { name: 'プロジェクトに保存' })
      .getByRole('group', { name: '保存先' }),
  ).toContainText(`${priceProject}/models/model.yaml`)

  // 雛形のままの幅は骨組みが読む数と違うので, 直す場所を出して書かない
  const file = `${priceProject}/models/model.yaml`
  await expectContract(dialog, file, [
    'inputs.x: 入力 x の最後の大きさを特徴の列と値段の列の数 3 にしてください (今は 17)',
  ])

  // 記録のデータセットから入力と出力を作ると, 入力の幅を特徴の 2 列と値段の 1 列の 3 にし
  // (売れた数, 原価, 商品の id, 行の id の列は数えない), 出力 units の値の数 1 はそのまま
  const data = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await expect(data.getByLabel('データセットのディレクトリ')).toHaveValue(
    `${priceProject}/outputs/dataset`,
  )
  await data.getByRole('button', { name: '入力と出力を作る' }).click()
  const result = data.getByRole('status')
  await expect(result).toContainText(
    'データセットから入れました: 入力 x [4, 3], タスク price, 目的の列 units, 出す値 1',
  )
  await expect(result).toContainText(
    '問題の設定 e2e の値段の見直し が値段を変える判断なので, ' +
      '入力の幅を, 商品の特徴の列と値段の列を合わせた 3 列にしました. ' +
      '値段 (商品の売り値) の列 price は, 入力の最後の列です. ' +
      '次の列は入力に入れません: 売れた数 (予測する値) の列 units, ' +
      '原価 (仕入れか作るのにかかる値段. 利益の計算に使う) の列 unit_cost, ' +
      '商品の id (値段を選ぶ単位) の列 product_id, 行を見分ける id の列 record_id',
  )
  await expect(result).toContainText(
    '定義は雛形 tabular_price のままなので, データに合わせる引数を ML-ModelControl の書き換え',
  )
  await expect(result).toContainText(
    '雛形の引数 features (特徴の数) を 17 から 3 にしました (入力 x の形の 2 番目, units1 の in_features)',
  )
  await expect(result).toContainText(
    '出力の値の数は変えていません (売れた数の予測は 1 行に 1 つの値なので, 雛形の 1 のまま使います)',
  )

  // 作った定義は骨組みに合うので, 合わない所を出さずに書く
  await expectSaved(page, dialog, file)
  await dialog.getByRole('button', { name: '閉じる' }).click()

  // 隠れ層の幅は変えず, 出力は (バッチ, 1) (誤り 0 件は確認の結果が届く前にも出るので, 入れた
  // 後の形を先に待つ)
  await expect(page.locator('[data-id="units1"]')).toContainText('[4, 32]', {
    timeout: 20_000,
  })
  await expect(page.locator('[data-id="units_softplus"]')).toContainText('[4, 1]')
  await expect(page.locator('header')).toContainText('誤り 0 件')
  await expect(page.getByRole('button', { name: '保存 *' })).toBeVisible()
})

test('時間をまたぐ量の判断の問題の設定を新しく始めるときに選ぶと需要の雛形を開き, 入力の幅と出力の値の数を学ぶ水準の数にする', async ({
  page,
}) => {
  await page.goto('/')
  const dialog = await openDialog(page)
  const demand = dialog
    .getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
    .getByRole('button', { name: /e2e の発注の見直し/ })
  await expect(demand).toContainText('雛形: tabular_demand')

  // 新規の画面で選ぶと雛形を開き, 保存先は記録の場所
  await demand.click()
  const note = dialog.getByRole('region', { name: '受け持ちに合う雛形' })
  await expect(note.getByRole('status')).toContainText(
    '問題の設定 e2e の発注の見直し の受け持ちに合う雛形 tabular_demand を開きました',
  )
  await expect(note.getByRole('button')).toHaveCount(0)
  await expect(
    dialog
      .getByRole('region', { name: 'プロジェクトに保存' })
      .getByRole('group', { name: '保存先' }),
  ).toContainText(`${demandProject}/models/model.yaml`)

  // 雛形のままの幅と水準の数は骨組みが読む数と違うので, 直す場所を出して書かない
  const file = `${demandProject}/models/model.yaml`
  await expectContract(dialog, file, [
    'inputs.x: 入力 x の最後の大きさを学習に使う列の数 5 にしてください (今は 16)',
    'outputs.quantiles: 出力 quantiles の最後の大きさを学ぶ水準の数 4 にしてください (今は 9)',
  ])

  // 記録のデータセットから入力と出力を作ると, 入力の幅を学習に使う 5 列にし, 出力 quantiles
  // の値の数を学習の設定の学ぶ水準 (答えた 0.1 と 0.9 に, 0.5 と水準 τ 0.75 を足したもの) の数 4 にする
  const data = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
  await expect(data.getByLabel('データセットのディレクトリ')).toHaveValue(
    `${demandProject}/outputs/dataset`,
  )
  await data.getByRole('button', { name: '入力と出力を作る' }).click()
  const result = data.getByRole('status')
  await expect(result).toContainText(
    'データセットから入れました: 入力 x [4, 5], タスク demand, 目的の列 demand, 出す値 4',
  )
  await expect(result).toContainText(
    '問題の設定 e2e の発注の見直し が時間をまたぐ量の判断なので, ' +
      '入力の幅を, 学習に使う 5 列にしました ' +
      '(商品の特徴, 過去の売れた数のずらしと移動平均, 曜日などの列). ' +
      '次の列は入力に入れません: ' +
      '日ごとの売れた数 (先の期間の合計が予測する値) の列 units, ' +
      '商品の id (発注の数を決める単位) の列 product_id, 日付 (発注を決める日) の列 date, ' +
      '在庫の数 (発注の数から引く) の列 stock, 行を見分ける id の列 record_id',
  )
  await expect(result).toContainText(
    '定義は雛形 tabular_demand のままなので, データに合わせる引数を ML-ModelControl の書き換え',
  )
  await expect(result).toContainText(
    '雛形の引数 features (特徴の数) を 16 から 5 にしました (入力 x の形の 2 番目, demand1 の in_features)',
  )
  await expect(result).toContainText(
    '出力 quantiles の値の数を, 学ぶ分位点の水準 0.1, 0.5, 0.75, 0.9 の数 4 ' +
      '(プロジェクトの学習の設定の水準. 水準ごとに 1 つの値) にしました ' +
      '(雛形の引数 levels を 9 から 4 に: demand2 の out_features)',
  )

  // 作った定義は骨組みに合うので, 合わない所を出さずに書く
  await expectSaved(page, dialog, file)
  await dialog.getByRole('button', { name: '閉じる' }).click()

  // 隠れ層の幅は変えず, 出力は (バッチ, 水準の数) (誤り 0 件は確認の結果が届く前にも出るので,
  // 入れた後の形を先に待つ)
  await expect(page.locator('[data-id="demand1"]')).toContainText('[4, 32]', {
    timeout: 20_000,
  })
  await expect(page.locator('[data-id="demand_cumsum"]')).toContainText('[4, 4]')
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

  // 雛形のままの利用者と品物の数は骨組みが読む数と違うので, 直す場所を出して書かない
  const file = `${rankProject}/models/model.yaml`
  await expectContract(dialog, file, [
    'inputs.user: 入力 user の最後の大きさを学習用の利用者の数 6 にしてください (今は 12)',
    'inputs.item: 入力 item の最後の大きさを学習用の品物の数 4 にしてください (今は 20)',
  ])

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
  await expect(result).toContainText(
    '定義は雛形 two_tower_retrieval のままなので, データに合わせる引数を ML-ModelControl の書き換え',
  )
  await expect(result).toContainText(
    '雛形の引数 users (利用者の数) を 12 から 6 にしました (入力 user の形の 2 番目, user_fc1 の in_features)',
  )
  await expect(result).toContainText(
    '雛形の引数 items (品物の数) を 20 から 4 にしました (入力 item の形の 2 番目, item_fc1 の in_features)',
  )

  // 作った定義は骨組みに合うので, 合わない所を出さずに書く
  await expectSaved(page, dialog, file)
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
