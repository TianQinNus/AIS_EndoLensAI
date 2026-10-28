import { useEffect, useRef } from 'react'
import { assetUrl, type FrameResult } from '../utils/endpoints'

export function FrameOverlay({ frame }: { frame: FrameResult }) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let cancelled = false
    const target = canvas.current
    if (!target || !frame.source_frame_url) return
    target.width = frame.width
    target.height = frame.height
    const ctx = target.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, target.width, target.height)
    const load = (url: string) =>
      new Promise<HTMLImageElement>((resolve, reject) => {
        const image = new Image()
        image.crossOrigin = 'anonymous'
        image.onload = () => resolve(image)
        image.onerror = () => reject(new Error('Image failed'))
        image.src = assetUrl(url) ?? url
      })
    void (async () => {
      const source = await load(frame.source_frame_url!)
      if (cancelled) return
      ctx.drawImage(source, 0, 0)
      const maskUrl = frame.edited_mask_url ?? frame.mask_url
      if (frame.status !== 'completed' || !maskUrl) return
      const mask = await load(maskUrl)
      if (cancelled) return
      const layer = document.createElement('canvas')
      layer.width = frame.width
      layer.height = frame.height
      const layerContext = layer.getContext('2d')
      if (!layerContext) return
      layerContext.drawImage(mask, 0, 0)
      const rgba = layerContext.getImageData(0, 0, layer.width, layer.height)
      for (let i = 0; i < rgba.data.length; i += 4) {
        const foreground = rgba.data[i] > 127
        rgba.data[i] = 0
        rgba.data[i + 1] = 240
        rgba.data[i + 2] = 110
        rgba.data[i + 3] = foreground ? 85 : 0
      }
      layerContext.putImageData(rgba, 0, 0)
      ctx.drawImage(layer, 0, 0)
    })().catch((error) => {
      if (!cancelled) console.error('Frame overlay failed', error)
    })
    return () => {
      cancelled = true
    }
  }, [
    frame.frame_id,
    frame.mask_url,
    frame.edited_mask_url,
    frame.source_frame_url,
    frame.width,
    frame.height,
    frame.status,
  ])

  return (
    <figure className="m-0">
      <figcaption className="sr-only">
        {frame.edited_mask_url ? 'Human-reviewed mask' : 'Original AI mask'}
      </figcaption>
      <canvas
        ref={canvas}
        aria-label="Saved frame with segmentation overlay"
        className="block h-auto max-h-full w-full object-contain"
      />
    </figure>
  )
}
