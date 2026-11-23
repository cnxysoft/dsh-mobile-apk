/**
 * Long-session history conversion caching for deep-frozen Harness messages.
 *
 * In long DSH sessions (tens of thousands of messages), every turn replays the
 * same deep-frozen Message prefix through:
 *   1. dsh-llm-pi-ai: assertSupportedHistory, toPiAssistant (readReplayState +
 *      JSON.parse(toolCall.arguments)), userContent (plain-text flattening).
 *   2. pi-ai/transform-messages: same-model assistant pass-through cloning.
 *   3. pi-ai/openai-responses-shared: readReasoningSignature (JSON.parse),
 *      parseTextSignature (JSON.parse), JSON.stringify(toolCall.arguments),
 *      sanitizeSurrogates (lookbehind regex over tens of MiB), and
 *      convertToolResultOutput double-filter allocations.
 *
 * Every cache is WeakMap/WeakSet-keyed on immutable (frozen) objects so entries
 * are automatically reclaimed on GC, never leak across sessions, and produce
 * bit-for-bit identical wire payloads.
 */
import { ENGINE_BASELINE, PINNED_PI_AI, assertEngineBaseline } from '../engine-versions.mjs'

export const LONG_SESSION_CACHE_MARKER = 'dsh-mobile long-session history cache'

const NM = 'usr/lib/node_modules/@deepseek-ai/dsh/node_modules/'
export const LONG_SESSION_CACHE_ADAPTER = NM + '@deepseek-ai/dsh-llm-pi-ai/lib/index.js'
export const LONG_SESSION_CACHE_TRANSFORM = NM + '@earendil-works/pi-ai/dist/api/transform-messages.js'
export const LONG_SESSION_CACHE_RESPONSES = NM + '@earendil-works/pi-ai/dist/api/openai-responses-shared.js'

export const LONG_SESSION_CACHE_FILES = Object.freeze([
  LONG_SESSION_CACHE_ADAPTER,
  LONG_SESSION_CACHE_TRANSFORM,
  LONG_SESSION_CACHE_RESPONSES,
])

function replaceOnceStrict(source, before, after, label) {
  const eol = source.includes('\r\n') ? '\r\n' : '\n'
  const b = before.replace(/\r?\n/g, eol)
  const a = after.replace(/\r?\n/g, eol)
  const count = source.split(b).length - 1
  if (count !== 1) throw new Error(`long-session-history-cache: anchor drift (${label}, matches=${count})`)
  return source.replace(b, a)
}

export function patchLongSessionAdapter(source) {
  if (source.includes('// ' + LONG_SESSION_CACHE_MARKER)) return source
  let out = source
  out = replaceOnceStrict(
    out,
    `function toPiAssistant(message, onDegrade, prefixDigests) {
\tconst source = message.source;
\tif (source.replayState === void 0) return foreignAssistant(message);
\ttry {
\t\treturn replayedAssistant(message, source, source.replayState, prefixDigests);
\t} catch (error) {`,
    `const __dshPiAssistantCache = /* @__PURE__ */ new WeakMap();
const __dshValidHistoryCache = /* @__PURE__ */ new WeakSet();
const __dshPlainUserContentCache = /* @__PURE__ */ new WeakMap();
function toPiAssistant(message, onDegrade, prefixDigests) {
\tconst frozen = Object.isFrozen(message);
\tif (frozen) {
\t\tconst cached = __dshPiAssistantCache.get(message);
\t\tif (cached !== void 0) {
\t\t\tif (cached.prefixDigest !== void 0) prefixDigests?.set(cached.assistant, cached.prefixDigest);
\t\t\treturn cached.assistant;
\t\t}
\t}
\tconst source = message.source;
\tif (source.replayState === void 0) {
\t\tconst assistant = foreignAssistant(message);
\t\tif (frozen) {
\t\t\tObject.freeze(assistant.content);
\t\t\tObject.freeze(assistant);
\t\t\t__dshPiAssistantCache.set(message, { assistant, prefixDigest: void 0 });
\t\t}
\t\treturn assistant;
\t}
\ttry {
\t\tconst assistant = replayedAssistant(message, source, source.replayState, prefixDigests);
\t\tif (frozen) {
\t\t\tconst prefixDigest = source.replayState?.response?.prefixDigest;
\t\t\tObject.freeze(assistant.content);
\t\t\tObject.freeze(assistant);
\t\t\t__dshPiAssistantCache.set(message, { assistant, prefixDigest });
\t\t}
\t\treturn assistant;
\t} catch (error) {`,
    'adapter.toPiAssistant'
  )
  out = replaceOnceStrict(
    out,
    `function assertSupportedHistory(messages) {
\tfor (const message of messages) {
\t\tif (message.role !== "developer" && message.content.some((block) => block.type === "tool-addition" || block.type === "tool-removal")) throw new LlmError("Tool-change blocks require developer role", "UNSUPPORTED_CONTENT");
\t\tif (message.role !== "user" && message.role !== "tool" && contentHasImage(message.content)) throw new LlmError(\`pi-ai cannot represent an image in an in-history \${message.role} message\`, "UNSUPPORTED_CONTENT");
\t\tif (message.role === "system" && message.content.some((block) => block.type !== "text")) throw new LlmError("pi-ai cannot represent non-text system messages", "UNSUPPORTED_CONTENT");
\t}
}`,
    `function assertSupportedHistory(messages) {
\tfor (const message of messages) {
\t\tif (__dshValidHistoryCache.has(message)) continue;
\t\tif (message.role !== "developer" && message.content.some((block) => block.type === "tool-addition" || block.type === "tool-removal")) throw new LlmError("Tool-change blocks require developer role", "UNSUPPORTED_CONTENT");
\t\tif (message.role !== "user" && message.role !== "tool" && contentHasImage(message.content)) throw new LlmError(\`pi-ai cannot represent an image in an in-history \${message.role} message\`, "UNSUPPORTED_CONTENT");
\t\tif (message.role === "system" && message.content.some((block) => block.type !== "text")) throw new LlmError("pi-ai cannot represent non-text system messages", "UNSUPPORTED_CONTENT");
\t\tif (Object.isFrozen(message)) __dshValidHistoryCache.add(message);
\t}
}`,
    'adapter.assertSupportedHistory'
  )
  out = replaceOnceStrict(
    out,
    `function userContent(blocks, requestImages, resolveImageAccess) {
\tconst content = [];`,
    `function userContent(blocks, requestImages, resolveImageAccess) {
\tconst cachedText = __dshPlainUserContentCache.get(blocks);
\tif (cachedText !== void 0) return cachedText;
\tconst content = [];`,
    'adapter.userContent.head'
  )
  out = replaceOnceStrict(
    out,
    `\tif (content.every((block) => block.type === "text")) return content.map((block) => block.text).join("");
\treturn content;`,
    `\tif (content.every((block) => block.type === "text")) {
\t\tconst joined = content.map((block) => block.text).join("");
\t\tif (Object.isFrozen(blocks)) __dshPlainUserContentCache.set(blocks, joined);
\t\treturn joined;
\t}
\treturn content;`,
    'adapter.userContent.tail'
  )
  return out + '\n// ' + LONG_SESSION_CACHE_MARKER + '\n'
}

export function patchLongSessionTransform(source) {
  if (source.includes('// ' + LONG_SESSION_CACHE_MARKER)) return source
  let out = source
  out = replaceOnceStrict(
    out,
    `export function transformMessages(messages, model, normalizeToolCallId) {`,
    `const __dshTransformSameModelPass = new WeakSet();
export function transformMessages(messages, model, normalizeToolCallId) {`,
    'transform.header'
  )
  out = replaceOnceStrict(
    out,
    `        if (msg.role === "assistant") {
            const assistantMsg = msg;
            const isSameModel = assistantMsg.provider === model.provider &&
                assistantMsg.api === model.api &&
                assistantMsg.model === model.id;
            const transformedContent = assistantMsg.content.flatMap((block) => {`,
    `        if (msg.role === "assistant") {
            const assistantMsg = msg;
            const isSameModel = assistantMsg.provider === model.provider &&
                assistantMsg.api === model.api &&
                assistantMsg.model === model.id;
            if (isSameModel && __dshTransformSameModelPass.has(assistantMsg)) {
                return assistantMsg;
            }
            const transformedContent = assistantMsg.content.flatMap((block) => {`,
    'transform.fastpath'
  )
  out = replaceOnceStrict(
    out,
    `            return {
                ...assistantMsg,
                content: transformedContent,
            };`,
    `            if (transformedContent.length === assistantMsg.content.length) {
                let same = true;
                for (let k = 0; k < transformedContent.length; k++) {
                    if (transformedContent[k] !== assistantMsg.content[k]) { same = false; break; }
                }
                if (same) {
                    if (isSameModel && Object.isFrozen(assistantMsg)) __dshTransformSameModelPass.add(assistantMsg);
                    return assistantMsg;
                }
            }
            return {
                ...assistantMsg,
                content: transformedContent,
            };`,
    'transform.preserveRef'
  )
  return out + '\n// ' + LONG_SESSION_CACHE_MARKER + '\n'
}

export function patchLongSessionResponses(source) {
  if (source.includes('// ' + LONG_SESSION_CACHE_MARKER)) return source
  let out = source
  out = replaceOnceStrict(
    out,
    `function convertToolResultOutput(model, content) {
    const textResult = content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\\n");
    const images = content.filter((c) => c.type === "image");
    const hasText = textResult.length > 0;
    if (images.length === 0 || !model.input.includes("image")) {
        return sanitizeSurrogates(hasText ? textResult : images.length > 0 ? "(see attached image)" : "(no tool output)");
    }
    const output = [];
    if (hasText) {
        output.push({ type: "input_text", text: sanitizeSurrogates(textResult) });
    }`,
    `const __dshResponsesAsstCache = new WeakMap();
function dshFastSanitizeSurrogates(text) {
    return typeof text.isWellFormed === "function" && text.isWellFormed() ? text : sanitizeSurrogates(text);
}
function convertToolResultOutput(model, content) {
    if (content.length === 1 && content[0].type === "text") {
        const singleText = content[0].text;
        return dshFastSanitizeSurrogates(singleText.length > 0 ? singleText : "(no tool output)");
    }
    const textResult = content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\\n");
    const images = content.filter((c) => c.type === "image");
    const hasText = textResult.length > 0;
    if (images.length === 0 || !model.input.includes("image")) {
        return dshFastSanitizeSurrogates(hasText ? textResult : images.length > 0 ? "(see attached image)" : "(no tool output)");
    }
    const output = [];
    if (hasText) {
        output.push({ type: "input_text", text: dshFastSanitizeSurrogates(textResult) });
    }`,
    'responses.toolResultOutput'
  )
  out = replaceOnceStrict(
    out,
    `                if (text.length > 0) {
                    messages.push({ role: instructionRole, content: sanitizeSurrogates(text) });
                }
            }
        }
        else if (msg.role === "user") {
            if (typeof msg.content === "string") {
                messages.push({
                    role: "user",
                    content: [{ type: "input_text", text: sanitizeSurrogates(msg.content) }],
                });
            }
            else {
                const content = msg.content.map((item) => {
                    if (item.type === "text") {
                        return {
                            type: "input_text",
                            text: sanitizeSurrogates(item.text),
                        };
                    }`,
    `                if (text.length > 0) {
                    messages.push({ role: instructionRole, content: dshFastSanitizeSurrogates(text) });
                }
            }
        }
        else if (msg.role === "user") {
            if (typeof msg.content === "string") {
                messages.push({
                    role: "user",
                    content: [{ type: "input_text", text: dshFastSanitizeSurrogates(msg.content) }],
                });
            }
            else {
                const content = msg.content.map((item) => {
                    if (item.type === "text") {
                        return {
                            type: "input_text",
                            text: dshFastSanitizeSurrogates(item.text),
                        };
                    }`,
    'responses.userSanitize'
  )
  out = replaceOnceStrict(
    out,
    `        else if (msg.role === "assistant") {
            const output = [];
            const assistantMsg = msg;
            const isSameProviderAndApi = assistantMsg.provider === model.provider && assistantMsg.api === model.api;
            const isSameModel = isSameProviderAndApi && assistantMsg.model === model.id;
            const isDifferentModel = isSameProviderAndApi && assistantMsg.model !== model.id;
            let textBlockIndex = 0;`,
    `        else if (msg.role === "assistant") {
            const assistantMsg = msg;
            const isSameProviderAndApi = assistantMsg.provider === model.provider && assistantMsg.api === model.api;
            const isSameModel = isSameProviderAndApi && assistantMsg.model === model.id;
            const isDifferentModel = isSameProviderAndApi && assistantMsg.model !== model.id;
            const hasGrammar = (options?.grammarToolInputProperties?.size ?? 0) > 0;
            const cacheKey = hasGrammar || !Object.isFrozen(assistantMsg) ? null : (model.provider + "\\0" + model.api + "\\0" + model.id);
            const cachedAsst = cacheKey !== null ? __dshResponsesAsstCache.get(assistantMsg) : undefined;
            if (cachedAsst !== undefined && cachedAsst.key === cacheKey) {
                if (cachedAsst.items.length === 0)
                    continue;
                if (options?.configurationUpdateEffort !== undefined &&
                    isSameModel &&
                    msg.providerThinkingLevel !== undefined) {
                    pushConfigurationUpdate(msg.providerThinkingLevel);
                }
                for (let ci = 0; ci < cachedAsst.items.length; ci++)
                    messages.push(cachedAsst.items[ci]);
                if (!isLeadingSystemMessage)
                    msgIndex++;
                continue;
            }
            const output = [];
            let usedFallbackId = false;
            let textBlockIndex = 0;`,
    'responses.assistantHead'
  )
  out = replaceOnceStrict(
    out,
    `                    if (!msgId) {
                        msgId = fallbackMessageId;
                    }
                    else if (msgId.length > 64) {
                        msgId = \`msg_\${shortHash(msgId)}\`;
                    }
                    output.push({
                        type: "message",
                        role: "assistant",
                        content: [{ type: "output_text", text: sanitizeSurrogates(textBlock.text), annotations: [] }],`,
    `                    if (!msgId) {
                        msgId = fallbackMessageId;
                        usedFallbackId = true;
                    }
                    else if (msgId.length > 64) {
                        msgId = \`msg_\${shortHash(msgId)}\`;
                    }
                    output.push({
                        type: "message",
                        role: "assistant",
                        content: [{ type: "output_text", text: dshFastSanitizeSurrogates(textBlock.text), annotations: [] }],`,
    'responses.assistantText'
  )
  out = replaceOnceStrict(
    out,
    `            if (output.length === 0)
                continue;
            if (options?.configurationUpdateEffort !== undefined &&
                isSameModel &&
                msg.providerThinkingLevel !== undefined) {
                pushConfigurationUpdate(msg.providerThinkingLevel);
            }
            messages.push(...reasonFirst(output));`,
    `            if (cacheKey !== null && !usedFallbackId) {
                const ordered = output.length === 0 ? output : reasonFirst(output);
                __dshResponsesAsstCache.set(assistantMsg, { key: cacheKey, items: ordered });
                if (ordered.length === 0)
                    continue;
                if (options?.configurationUpdateEffort !== undefined &&
                    isSameModel &&
                    msg.providerThinkingLevel !== undefined) {
                    pushConfigurationUpdate(msg.providerThinkingLevel);
                }
                for (let ci = 0; ci < ordered.length; ci++)
                    messages.push(ordered[ci]);
                if (!isLeadingSystemMessage)
                    msgIndex++;
                continue;
            }
            if (output.length === 0)
                continue;
            if (options?.configurationUpdateEffort !== undefined &&
                isSameModel &&
                msg.providerThinkingLevel !== undefined) {
                pushConfigurationUpdate(msg.providerThinkingLevel);
            }
            messages.push(...reasonFirst(output));`,
    'responses.assistantTail'
  )
  return out + '\n// ' + LONG_SESSION_CACHE_MARKER + '\n'
}

const PATCHERS = [
  patchLongSessionAdapter,
  patchLongSessionTransform,
  patchLongSessionResponses,
]

const CONTRACTS = [
  ['__dshPiAssistantCache', '__dshValidHistoryCache', '__dshPlainUserContentCache'],
  ['__dshTransformSameModelPass'],
  ['__dshResponsesAsstCache', 'dshFastSanitizeSurrogates'],
]

export function planLongSessionCache(read) {
  const adapterPkg = JSON.parse(read(NM + '@deepseek-ai/dsh-llm-pi-ai/package.json'))
  if (adapterPkg.name !== '@deepseek-ai/dsh-llm-pi-ai') {
    throw new Error('long-session-history-cache: adapter package name mismatch')
  }
  assertEngineBaseline(adapterPkg, 'long-session-history-cache @deepseek-ai/dsh-llm-pi-ai')

  const piPkg = JSON.parse(read(NM + '@earendil-works/pi-ai/package.json'))
  if (piPkg.name !== '@earendil-works/pi-ai' || piPkg.version !== PINNED_PI_AI) {
    throw new Error(`long-session-history-cache requires @earendil-works/pi-ai@${PINNED_PI_AI} (got ${piPkg.name}@${piPkg.version})`)
  }

  return LONG_SESSION_CACHE_FILES.map((file, index) => {
    const before = read(file)
    if (before.includes('// ' + LONG_SESSION_CACHE_MARKER)) {
      if (!CONTRACTS[index].every(token => before.includes(token))) {
        throw new Error('long-session-history-cache: incomplete installed patch: ' + file)
      }
      return { file, before, after: before }
    }
    return { file, before, after: PATCHERS[index](before) }
  })
}
