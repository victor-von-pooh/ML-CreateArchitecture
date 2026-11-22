import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { expect, test, type Page } from '@playwright/test'

// e2e/serve.sh が ML-EDA と ML-Quest で作った, 書き出したデータセット (3 列の
// 3 クラスの分類) と学習のプロジェクトの骨組み
const dataset = fileURLToPath(new URL('.work/dataset', import.meta.url))
// 灰色の画像と背景を含む 3 クラスの領域のファイルの組 (版 3. 高さと幅は決まらない)
const masks = fileURLToPath(new URL('.work/masks', import.meta.url))
const project = fileURLToPath(new URL('.work/project', import.meta.url))
const modelFile = `${project}/models/model.yaml`

// プロジェクトへの保存は 2 回目に上書きを確かめるので, 書いた順に実行
test.describe.configure({ mode: 'serial' })

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
