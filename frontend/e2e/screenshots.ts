/// <reference lib="dom" />
// README の画面の例 (docs/images/*.png) を撮るスクリプト (CI では動かさない).
// 先に別の端末で pnpm build と bash e2e/serve.sh を動かし, frontend で
// node e2e/screenshots.ts を実行する. 作業用の置き場の場所は ~ で示す
import { fileURLToPath } from 'node:url'

import { chromium, type Page } from '@playwright/test'

const port = Number(process.env.E2E_PORT ?? 8792)
const work = fileURLToPath(new URL('.work', import.meta.url))
const images = fileURLToPath(new URL('../../docs/images', import.meta.url))

/** 画面の文字と欄の値の作業用の置き場の場所を ~ に置き換える */
function hidePrefix(prefix: string) {
  const swap = (text: string) => text.split(prefix).join('~')
  const hide = () => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
    for (let node = walker.nextNode(); node; node = walker.nextNode())
      if (node.nodeValue?.includes(prefix)) node.nodeValue = swap(node.nodeValue)
    for (const input of Array.from(document.querySelectorAll('input')))
      if (input.value.includes(prefix)) input.value = swap(input.value)
  }
  document.addEventListener('DOMContentLoaded', () => {
    new MutationObserver(hide).observe(document.body, {
      subtree: true,
      childList: true,
      characterData: true,
    })
  })
}

async function shoot(page: Page, name: string) {
  await page.mouse.move(1590, 890)
  await page.waitForTimeout(800)
  await page.screenshot({ path: `${images}/${name}.png` })
}

/** 下書きの残らない新しい画面を開く */
async function fresh() {
  const context = await browser.newContext({
    baseURL: `http://127.0.0.1:${port}`,
    viewport: { width: 1600, height: 900 },
    colorScheme: 'light',
    extraHTTPHeaders: { 'x-createarchitecture-token': 'e2e-token' },
  })
  await context.addInitScript(hidePrefix, work)
  const page = await context.newPage()
  page.on('dialog', (dialog) => void dialog.accept())
  await page.goto('/')
  return page
}

/** つながりの順に並べ, 全体を画面に収めてから拡大 (負なら縮小) する */
async function arrange(page: Page, zoom: number) {
  await page.getByRole('button', { name: '整列' }).click()
  await page.locator('.react-flow__controls-fitview').click()
  const button = zoom > 0 ? 'zoomin' : 'zoomout'
  for (let i = 0; i < Math.abs(zoom); i++)
    await page.locator(`.react-flow__controls-${button}`).click()
  await page.waitForTimeout(500)
}

async function openExample(page: Page, name: string, node: string, shape: string) {
  await page.getByRole('button', { name: '開く', exact: true }).click()
  await page.getByRole('button', { name, exact: true }).click()
  await page.locator(`[data-id="${node}"]`).getByText(shape).waitFor({ timeout: 20_000 })
}

/** 全結合 (16 → out) を置いて入力と出力につなぐ */
async function placeLinear(page: Page, out: string) {
  await page.getByPlaceholder('部品を探す').fill('nn.Linear')
  await page
    .getByRole('button', { name: /^nn\.Linear/ })
    .first()
    .click()
  await page.getByLabel('in_features').fill('16')
  await page.getByLabel('in_features').press('Enter')
  await page.getByLabel('out_features').fill(out)
  await page.getByLabel('out_features').press('Enter')
  await page.getByLabel(/^input/).selectOption('x')
  await page.getByRole('button', { name: '出力' }).click()
  await page.getByLabel(/つなぎ先/).selectOption('linear1')
}

const browser = await chromium.launch({ channel: 'chrome' })
let page = await fresh()

// VAE の雛形で, 再パラメータ化のノードを選ぶ
await openExample(page, 'vae', 'z', '[8, 8]')
await arrange(page, 2)
await page.locator('[data-id="z"]').getByText('z', { exact: true }).click()
await shoot(page, 'vae')

// CNN の例で, 畳み込みのノードを選ぶ
page = await fresh()
await openExample(page, 'cnn_classifier', 'head', '[4, 10]')
await arrange(page, 0)
await page.locator('[data-id="conv1"]').getByText('conv1', { exact: true }).click()
await shoot(page, 'cnn')

// 開くの一覧 (保存した定義, 同梱の例と雛形, 学習の実験の定義の写し)
page = await fresh()
await page.getByRole('button', { name: '開く', exact: true }).click()
const open = page.getByRole('dialog', { name: '定義を開く' })
await open.getByRole('button', { name: 'vae', exact: true }).waitFor()
await shoot(page, 'open')

// 学習の実験の定義の写しを探す
const copies = open.getByRole('region', { name: '学習の実験の定義の写し' })
await copies.getByLabel('プロジェクトか問題の設定').fill(`${work}/project`)
await copies.getByRole('button', { name: '実験を探す' }).click()
const run = copies.getByRole('button', { name: /e2e_run/ })
await run.waitFor()
await run.scrollIntoViewIfNeeded()
await shoot(page, 'run_copy')

// 形の誤り (入る形 16 に対して in_features 8)
page = await fresh()
await placeLinear(page, '4')
await arrange(page, -3)
await page.locator('[data-id="linear1"]').click()
await page.getByLabel('in_features').fill('8')
await page.getByLabel('in_features').press('Enter')
await page.locator('header').getByText('誤り 1 件').waitFor({ timeout: 20_000 })
await page.locator('[data-id="linear1"]').click()
await shoot(page, 'error')

// データセットとプロジェクト: 問題の設定を選び, 入力と出力を作る
page = await fresh()
await page.getByRole('button', { name: 'データセットとプロジェクト' }).click()
const dialog = page.getByRole('dialog', { name: 'データセットとプロジェクト' })
await dialog
  .getByRole('list', { name: 'ML-Pathfinder に保存した問題の設定' })
  .getByRole('button', { name: /e2e の売上/ })
  .click()
const data = dialog.getByRole('region', { name: 'データセットから入力と出力を作る' })
await data.getByRole('button', { name: '入力と出力を作る' }).click()
await data
  .getByRole('status')
  .getByText(/^データセットから入れました/)
  .waitFor()
await dialog.getByRole('group', { name: '保存先' }).waitFor()
await data.evaluate((el) => el.scrollIntoView({ block: 'start' }))
await shoot(page, 'dataset')

// パラメータの状態: 凍結と外部の重みを書き, 読んだ数を確かめる
page = await fresh()
await placeLinear(page, '4')
await arrange(page, -3)
await page.locator('header').getByText('誤り 0 件').waitFor({ timeout: 20_000 })
await page.locator('[data-id="linear1"]').click()
const state = page.getByRole('group', { name: 'パラメータの状態' })
await state.getByLabel('学習する').uncheck()
await state.getByLabel('外部の重み (weights)').selectOption('path')
await state.getByLabel(/ファイルのパス/).fill('e2e_base.pt')
await state.getByLabel(/ファイルのパス/).press('Enter')
await state.getByRole('button', { name: '付け替えを足す' }).click()
const rule = state.getByRole('group', { name: '付け替え 1' })
await rule.getByLabel(/^前置き/).fill('model.fc.')
await rule.getByLabel(/^前置き/).press('Enter')
await state.getByLabel(/合わない重みがあれば止める/).uncheck()
await state.getByRole('button', { name: '重みを読んで確かめる' }).click()
const report = state.getByRole('status', { name: '重みの読み込みの報告' })
await report.getByText('linear1').waitFor({ timeout: 20_000 })
await report.scrollIntoViewIfNeeded()
await shoot(page, 'param_state')

await browser.close()
