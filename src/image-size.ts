/**
 * The size of an image, read from the image instead of from the request.
 *
 * A cloud provider is free to answer a smaller picture than it was asked for.
 * Pollinations does exactly that: a 1024×1024 request comes back as a
 * 768×768 JPEG. The providers used to report the width and height they had
 * asked for, so the gallery labelled the result 1024×1024 and the session
 * saved a size the file never had. The bytes are the only authority here.
 *
 * The parser understands the formats these providers actually return — PNG,
 * JPEG, WebP and GIF — and returns null for anything else, which leaves the
 * provider's claim in place rather than inventing a size.
 */

export interface ImageSize {
  width: number;
  height: number;
}

/** The dimensions in an encoded image, or null when the header is not one we know. */
export function imageDimensions(bytes: Buffer): ImageSize | null {
  // PNG: signature, then the IHDR chunk whose first two fields are the size.
  if (
    bytes.length >= 24 &&
    bytes.subarray(0, 8).toString('hex') === '89504e470d0a1a0a' &&
    bytes.toString('latin1', 12, 16) === 'IHDR'
  ) {
    return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
  }

  // JPEG: walk the marker segments to the start-of-frame header. The size is
  // two bytes each, after the precision byte. APPn and DQT segments come
  // first and are skipped by their own length.
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2;
    while (offset + 9 <= bytes.length) {
      if (bytes[offset] !== 0xff) {
        offset += 1;
        continue;
      }
      const marker = bytes[offset + 1];
      // Fill bytes and markers with no length field.
      if (marker === 0xff) {
        offset += 1;
        continue;
      }
      if (marker === 0x01 || marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7)) {
        offset += 2;
        continue;
      }
      if (marker === 0xd9) break;
      const length = bytes.readUInt16BE(offset + 2);
      if (length < 2) break;
      const isStartOfFrame =
        marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
      if (isStartOfFrame) {
        return { height: bytes.readUInt16BE(offset + 5), width: bytes.readUInt16BE(offset + 7) };
      }
      offset += 2 + length;
    }
    return null;
  }

  // WebP has three headers depending on the encoding.
  if (
    bytes.length >= 30 &&
    bytes.toString('latin1', 0, 4) === 'RIFF' &&
    bytes.toString('latin1', 8, 12) === 'WEBP'
  ) {
    const kind = bytes.toString('latin1', 12, 16);
    if (kind === 'VP8X') {
      return { width: 1 + bytes.readUIntLE(24, 3), height: 1 + bytes.readUIntLE(27, 3) };
    }
    if (kind === 'VP8 ' && bytes.length >= 30) {
      return {
        width: bytes.readUInt16LE(26) & 0x3fff,
        height: bytes.readUInt16LE(28) & 0x3fff,
      };
    }
    if (kind === 'VP8L' && bytes.length >= 25) {
      const bits = bytes.readUInt32LE(21);
      return { width: (bits & 0x3fff) + 1, height: ((bits >> 14) & 0x3fff) + 1 };
    }
  }

  if (bytes.length >= 10 && bytes.toString('latin1', 0, 3) === 'GIF') {
    return { width: bytes.readUInt16LE(6), height: bytes.readUInt16LE(8) };
  }

  return null;
}

/**
 * The size to report for a provider's answer.
 *
 * Falls back to the provider's claim only when the bytes are not a format the
 * parser knows. An empty payload keeps the claim too, because the server
 * stores the bytes before it empties the data field.
 */
export function actualImageSize(image: { data: string; width: number; height: number }): ImageSize {
  if (!image.data) return { width: image.width, height: image.height };
  return imageDimensions(Buffer.from(image.data, 'base64')) ?? {
    width: image.width,
    height: image.height,
  };
}
