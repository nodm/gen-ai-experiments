import * as fs from 'node:fs/promises';
import {Ollama} from '@langchain/ollama';
import config from './config.js';

export async function multimodalExample(
  question = "What's in this image?",
  imagePath = './src/assets/example.jpeg',
  options?: {
    modelName?: string;
    temperature?: number;
    maxRetries?: number;
  }
) {
  const imageData = await fs.readFile(imagePath);
  const multimodalLLm = new Ollama({
    model: options?.modelName ?? config.multimodalModelName,
    temperature: options?.temperature ?? config.temperature,
    maxRetries: options?.maxRetries ?? config.maxRetries,
  }).bind({
    images: [imageData.toString('base64')],
  });
  const res = await multimodalLLm.invoke(question);

  console.log(res);
}
