/** Sample TypeScript for AST parser smoke tests. */
import { Counter } from "./sample";

export interface Named {
  name: string;
}

export class Person implements Named {
  constructor(public name: string) {}

  hello(): string {
    return greet(this.name);
  }
}

export function greet(name: string): string {
  const counter = new Counter(1);
  counter.increment();
  return `Hi ${name}`;
}
