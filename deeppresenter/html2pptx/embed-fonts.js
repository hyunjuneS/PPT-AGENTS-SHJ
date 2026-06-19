'use strict';
/**
 * embed-fonts.js
 * Inject TTF/OTF font files into an already-generated PPTX buffer.
 *
 * PPTX is a ZIP. To embed a font PowerPoint actually uses, we need to:
 *   1. Add the font binary to  ppt/fonts/fontN.fntdata
 *   2. Register a relationship in  ppt/_rels/presentation.xml.rels
 *   3. Add <p:embeddedFont> to ppt/presentation.xml
 *   4. Register the content type in [Content_Types].xml
 */

const fs   = require('fs');
const path = require('path');
const JSZip = require('jszip');

/**
 * Read the Font Family Name (nameID=1) from a TTF/OTF binary buffer.
 * Prefers Windows platform (UTF-16BE), falls back to Mac (ASCII).
 * Returns null if the name table cannot be parsed.
 */
function readTtfFamilyName(buffer) {
  try {
    const numTables = buffer.readUInt16BE(4);
    for (let i = 0; i < numTables; i++) {
      const base = 12 + i * 16;
      if (buffer.toString('ascii', base, base + 4) !== 'name') continue;

      const tableOffset = buffer.readUInt32BE(base + 8);
      const count       = buffer.readUInt16BE(tableOffset + 2);
      const strBase     = tableOffset + buffer.readUInt16BE(tableOffset + 4);

      let best = null; // { priority, name }

      for (let j = 0; j < count; j++) {
        const r          = tableOffset + 6 + j * 12;
        const platformID = buffer.readUInt16BE(r);
        const nameID     = buffer.readUInt16BE(r + 6);
        const length     = buffer.readUInt16BE(r + 8);
        const offset     = buffer.readUInt16BE(r + 10);

        if (nameID !== 1) continue; // only Font Family Name

        const start = strBase + offset;

        if (platformID === 3 && (!best || best.priority < 3)) {
          // Windows / UTF-16BE
          const slice = Buffer.from(buffer.slice(start, start + length));
          slice.swap16();
          best = { priority: 3, name: slice.toString('utf16le') };
        } else if (platformID === 1 && (!best || best.priority < 1)) {
          // Mac / ASCII (MacRoman subset)
          best = { priority: 1, name: buffer.toString('ascii', start, start + length) };
        }
      }
      return best ? best.name : null;
    }
  } catch (_) {}
  return null;
}

/**
 * Scan `fontsDir` for *.ttf / *.otf files, embed each into `pptxBuffer`,
 * and return the modified PPTX as a Buffer.
 *
 * @param {Buffer} pptxBuffer
 * @param {string} fontsDir   - directory containing TTF/OTF files
 * @returns {Promise<Buffer>}
 */
async function embedFontsInPptx(pptxBuffer, fontsDir) {
  const fontFiles = fs.readdirSync(fontsDir)
    .filter(f => /\.(ttf|otf)$/i.test(f));

  if (fontFiles.length === 0) return pptxBuffer;

  const zip = await JSZip.loadAsync(pptxBuffer);

  // ── read existing XML ─────────────────────────────────────────────────────
  const ctXml     = await zip.file('[Content_Types].xml').async('string');
  const presXml   = await zip.file('ppt/presentation.xml').async('string');
  const relsPath  = 'ppt/_rels/presentation.xml.rels';
  const relsXml   = await zip.file(relsPath).async('string');

  // find highest existing rId number
  const existingIds = [...relsXml.matchAll(/Id="rId(\d+)"/g)].map(m => parseInt(m[1]));
  let nextId = existingIds.length ? Math.max(...existingIds) + 1 : 1;

  let addCT   = '';   // additions to [Content_Types].xml
  let addRels = '';   // additions to presentation.xml.rels
  let addFont = '';   // additions to embeddedFontLst

  for (const fileName of fontFiles) {
    const filePath   = path.join(fontsDir, fileName);
    const ttfBuffer  = fs.readFileSync(filePath);
    const familyName = readTtfFamilyName(ttfBuffer)
                       || path.basename(fileName, path.extname(fileName));
    const rId        = `rId${nextId}`;
    const fontEntry  = `font${nextId}.fntdata`;
    const partName   = `/ppt/fonts/${fontEntry}`;

    zip.file(`ppt/fonts/${fontEntry}`, ttfBuffer);

    addCT   += `\n  <Override PartName="${partName}" ContentType="application/x-fontdata"/>`;
    addRels += `\n  <Relationship Id="${rId}"`
             + ` Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/font"`
             + ` Target="fonts/${fontEntry}"/>`;
    addFont += `\n    <p:embeddedFont>`
             + `<p:font typeface="${familyName}"/>`
             + `<p:regular r:id="${rId}"/>`
             + `</p:embeddedFont>`;

    console.log(`[embed-fonts] ${familyName}  ←  ${fileName}`);
    nextId++;
  }

  // ── patch [Content_Types].xml ─────────────────────────────────────────────
  zip.file('[Content_Types].xml', ctXml.replace('</Types>', `${addCT}\n</Types>`));

  // ── patch presentation.xml.rels ───────────────────────────────────────────
  zip.file(relsPath, relsXml.replace('</Relationships>', `${addRels}\n</Relationships>`));

  // ── patch presentation.xml — insert / extend <p:embeddedFontLst> ─────────
  let updatedPres;
  if (presXml.includes('<p:embeddedFontLst>')) {
    updatedPres = presXml.replace(
      '</p:embeddedFontLst>',
      `${addFont}\n  </p:embeddedFontLst>`,
    );
  } else {
    updatedPres = presXml.replace(
      '</p:presentation>',
      `  <p:embeddedFontLst>${addFont}\n  </p:embeddedFontLst>\n</p:presentation>`,
    );
  }
  zip.file('ppt/presentation.xml', updatedPres);

  return zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' });
}

module.exports = { embedFontsInPptx, readTtfFamilyName };
