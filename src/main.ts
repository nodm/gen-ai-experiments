import {multimodalExample} from './multi-modal.js';
import {streamingExample} from './streaming.js';
import {chatbot} from './chatbot.js';
import {ragQuickStartGuide} from './rag-quick-start-guide.js';

export async function main() {
  console.time('LLM call');

  await streamingExample();

  await multimodalExample(
    "What are the PC's brand and model? It looks like the Apple iMac",
    './src/assets/example.jpeg',
    {temperature: 0.2}
  );

  await chatbot();

  await ragQuickStartGuide();

  console.timeEnd('LLM call');
}
