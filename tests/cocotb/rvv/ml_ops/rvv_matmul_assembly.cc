#include <riscv_vector.h>
#include <stdint.h>

constexpr size_t kLhsRows = 32;
constexpr size_t kRhsCols = 32;
constexpr size_t kInner = 128;

int8_t lhs_input[kLhsRows * kInner] __attribute__((section(".data")))
__attribute__((aligned(16)));
int8_t rhs_input[kInner * kRhsCols] __attribute__((section(".data")))
__attribute__((aligned(16)));
int32_t result_output[kLhsRows * kRhsCols] __attribute__((section(".data")))
__attribute__((aligned(16)));

// Assume rhs is column major.
void MatMul(size_t lhs_rows, size_t inner, size_t rhs_cols, const int8_t* lhs,
            const int8_t* rhs, int32_t* result) {
  size_t vlmax;
  asm volatile("vsetvli %0, %1, e8, m2, ta, ma" : "=r"(vlmax) : "r"(inner));

  for (size_t r = 0; r < lhs_rows; r++) {
    int32_t* result_row = result + (r * rhs_cols);

    for (size_t c = 0; c < rhs_cols; c++) {
      const int8_t* lhs_row = lhs + (r * inner);
      const int8_t* rhs_col = rhs + (c * inner);
      size_t k = inner;

      // Initialize v16 accumulator (e32m8) and v1 scalar reduction source to
      // zero
      asm volatile(
          "vsetvli zero, %0, e32, m8, ta, ma;\n\t"
          "vmv.v.i v16, 0;\n\t"
          "vsetivli zero, 1, e32, m1, ta, ma;\n\t"
          "vmv.v.i v1, 0"
          :
          : "r"(vlmax));

      while (k > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e8, m2, ta, ma" : "=r"(vl) : "r"(k));

        // Load inputs (v4, v6), widen to e16m4 (v8, v12), multiply-accumulate
        // into v16 (e32m8)
        asm volatile(
            "vsetvli zero, %0, e8, m2, ta, ma;\n\t"
            "vle8.v v4, (%1);\n\t"
            "vle8.v v6, (%2);\n\t"
            "vwadd.vx v8, v4, zero;\n\t"
            "vwadd.vx v12, v6, zero;\n\t"
            "vsetvli zero, %0, e16, m4, ta, ma;\n\t"
            "vwmacc.vv v16, v8, v12"
            :
            : "r"(vl), "r"(lhs_row), "r"(rhs_col));

        lhs_row += vl;
        rhs_col += vl;
        k -= vl;
      }

      // Reduce v16 over vlmax into v1, store scalar result
      asm volatile(
          "vsetvli zero, %0, e32, m8, ta, ma;\n\t"
          "vredsum.vs v1, v16, v1;\n\t"
          "vsetivli zero, 1, e32, m1, ta, ma;\n\t"
          "vse32.v v1, (%1)"
          :
          : "r"(vlmax), "r"(result_row + c));
    }
  }
}

int main() {
  MatMul(kLhsRows, kInner, kRhsCols, lhs_input, rhs_input, result_output);
  return 0;
}