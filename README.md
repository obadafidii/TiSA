# TiSA

TiSA is an educational tensor instruction-set architecture, compiler toolchain, and accelerator simulator built in C++.

The project is intentionally split into two layers:

1. **Working target stack today:** TinyTensor source / TISA assembly -> TISA bytecode -> tensor VM/simulator.
2. **MLIR integration scaffold:** an out-of-tree `tisa` dialect and `tisa-opt` driver, ready for lowering/serialization passes.

The first layer is fully runnable and tested without LLVM/MLIR installed. The MLIR subtree requires an MLIR development build/install and is kept optional.

## 1. Build the working runtime/compiler

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

### Compile TinyTensor source

```bash
./build/tiny-tensorc examples/matmul.tt /tmp/matmul.tbc
./build/tisa-run /tmp/matmul.tbc --trace
```

Expected tensor output:

```text
tensor[2x2]
  [0, 22]
  [43, 0]
```

### Assemble TISA directly

```bash
./build/tisa-as examples/matmul.tasm /tmp/matmul.tbc
./build/tisa-run /tmp/matmul.tbc
```

## 2. TinyTensor source language

```text
tensor A [2,2] = 1, 2, 3, 4
tensor B [2,2] = 5, 6, 7, 8
C = matmul A B
D = relu C
print D
```

`tiny-tensorc` assigns virtual tensor registers and emits the same bytecode that the future MLIR backend must emit.

## 3. TISA assembly

```text
.const A [2,2] = 1, 2, 3, 4
.const B [2,2] = 5, 6, 7, 8

const r0, A
const r1, B
matmul r2, r0, r1
relu r3, r2
print r3
halt
```

## 4. Build the MLIR dialect

You need an LLVM/MLIR build that exposes `MLIRConfig.cmake`.

```bash
cmake -S . -B build-mlir -G Ninja \
  -DTISA_BUILD_MLIR=ON \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm
cmake --build build-mlir --target tisa-opt
```

Then inspect the target dialect:

```bash
./build-mlir/mlir/tools/tisa-opt/tisa-opt mlir/examples/target.mlir
```

> The core C++ VM/compiler has been compiled and tested in the supplied environment. The optional MLIR subtree is a current-style out-of-tree scaffold, but it could not be compiled here because MLIR development libraries are not installed in this execution environment.

## 5. The actual MLIR project to implement next

The important pipeline is:

```text
PyTorch / tiny tensor program
        |
        v
Torch-MLIR / TOSA / Linalg
        |
        |  LowerToTISA pass
        v
      tisa.* MLIR
        |
        |  register assignment + serializer
        v
       .tbc
        |
        v
     tisa-run
```

Start with **Linalg or TOSA -> TISA**, not PyTorch directly. PyTorch import is a separate frontend problem.

### First lowering rules

- dense tensor constant -> `tisa.const`
- elementwise add -> `tisa.add`
- elementwise multiply -> `tisa.mul`
- supported matrix multiply -> `tisa.matmul`
- ReLU/clamp pattern -> `tisa.relu`

The pass must reject dynamic shapes, unsupported element types, broadcasting, and unsupported ranks in the MVP.

## 6. What you need to know

### Required first

- C++ ownership/value semantics, `std::vector`, RAII, CMake.
- Linear algebra: tensor rank/shape, elementwise operations, matrix multiplication.
- Computer architecture basics: ISA, opcode, register file, instruction encoding, memory vs registers.
- Compiler basics: AST, IR, lowering, legality, rewrite patterns, target-specific IR.
- MLIR basics: operations, SSA values, types/attributes, dialects, ODS/TableGen, pattern rewriting, dialect conversion.

### Learn after the MVP works

- Tensor layouts/strides.
- Bufferization (`tensor` -> `memref`).
- Tiling and loop nests.
- Vectorization/SIMD.
- Accelerator SRAM/global memory and explicit DMA.
- Cost models and scheduling.
- Quantization (`int8`, zero points, scale).
- Async execution / command queues.

## 7. Recommended milestone order

1. **ISA semantics** — DONE in this starter.
2. **Binary encoding/decoder** — DONE.
3. **Instruction-set simulator** — DONE.
4. **Control frontend** — DONE (`tiny-tensorc`).
5. **MLIR target dialect** — SCAFFOLDED.
6. **High-level tensor -> TISA conversion pass**.
7. **TISA MLIR -> bytecode translation**.
8. **Shape/type legality diagnostics**.
9. **Add local memory + load/store/DMA instructions**.
10. **Tile matrix multiplication**.
11. **Add a cycle/cost model to the simulator**.
12. **Import a tiny real model through TOSA/Torch-MLIR**.

## 8. Repository map

```text
include/tisa/       Runtime/compiler public API
src/                Tensor semantics, assembler, bytecode, compiler, VM
tools/tiny-tensorc  Tiny source-language compiler
tools/tisa-as       TISA assembler
tools/tisa-run      Instruction-set simulator
examples/           Runnable programs
mlir/               Optional custom MLIR dialect/tool scaffold
docs/architecture.md Architecture and target contract
```
