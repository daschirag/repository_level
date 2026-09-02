/** Sample JavaScript for AST parser smoke tests. */
import { readFile } from "fs";

export class Counter {
  constructor(start = 0) {
    this.value = start;
  }

  increment() {
    this.value = add(this.value, 1);
    return this.value;
  }
}

export function add(a, b) {
  return a + b;
}

export const double = (n) => add(n, n);

readFile("package.json", () => {});
