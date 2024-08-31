import {ChatPromptTemplate} from '@langchain/core/prompts';
import {StringOutputParser} from '@langchain/core/output_parsers';
import {Ollama} from '@langchain/ollama';
import config from './config.js';

export async function streamingExample(options?: {modelName?: string; temperature?: number; maxRetries?: number}) {
  const llm = new Ollama({
    model: options?.modelName ?? config.modelName,
    temperature: options?.temperature ?? config.temperature,
    maxRetries: options?.maxRetries ?? config.maxRetries,
  });

  // const prompt = ChatPromptTemplate.fromMessages([
  //   ['system', 'Translate the following into {language}:'],
  //   ['user', '{text}'],
  // ]);

  const prompt = ChatPromptTemplate.fromTemplate('How to say {text} in {language}:\n');

  // await prompt.formatMessages({
  //   output_language: 'German',
  //   input: 'I love programming.',
  // });

  const outputParser = new StringOutputParser();

  const chain = prompt.pipe(llm).pipe(outputParser);

  // const result = await chain.invoke({
  //   language: 'German',
  //   text: 'I love programming.',
  // });
  // console.log(result);

  const stream = await chain.stream({
    language: 'Lithuanian',
    text: 'I love programming.',
  });

  for await (const message of stream) {
    process.stdout.write(message);
  }
  process.stdout.write('\n');
}
