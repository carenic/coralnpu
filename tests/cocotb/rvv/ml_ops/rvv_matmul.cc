#include <riscv_vector.h>
#include <stdint.h>

constexpr size_t kLhsRows = 32;
constexpr size_t kRhsCols = 32;
constexpr size_t kInner = 128;

// mcontext0 val used in test for power period extraction
// mcontext0 is io_coralnpu_csr_value_8 in waveform
uint32_t mcontext0_write_value;

int8_t lhs_input[kLhsRows * kInner] __attribute__((section(".data")))
__attribute__((aligned(16)));
int8_t rhs_input[kInner * kRhsCols] __attribute__((section(".data")))
__attribute__((aligned(16)));
int32_t result_output[kLhsRows * kRhsCols] __attribute__((section(".data")))
__attribute__((aligned(16)));

// Assume rhs is column major.
void MatMul(size_t lhs_rows, size_t inner, size_t rhs_cols, const int8_t* lhs,
            const int8_t* rhs, int32_t* result) {
  size_t vlmax = __riscv_vsetvl_e8m2(inner);

  for (size_t r = 0; r < lhs_rows; r++) {
    int32_t* result_row = result + (r * rhs_cols);

    for (size_t c = 0; c < rhs_cols; c++) {
      const int8_t* lhs_row = lhs + (r * inner);
      const int8_t* rhs_col = rhs + (c * inner);
      vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vlmax);
      vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
      size_t k = inner;
      while (k) {
        size_t vl = __riscv_vsetvl_e8m2(k);

        vint8m2_t vrhs = __riscv_vle8_v_i8m2(rhs_col, vl);
        vint16m4_t vrhs16 = __riscv_vwadd_vx_i16m4(vrhs, 0, vl);
        rhs_col += vl;

        vint8m2_t vlhs = __riscv_vle8_v_i8m2(lhs_row, vl);
        vint16m4_t vlhs16 = __riscv_vwadd_vx_i16m4(vlhs, 0, vl);
        lhs_row += vl;

        vacc = __riscv_vwmacc_vv_i32m8(vacc, vlhs16, vrhs16, vl);
        k -= vl;
      }

      vint32m1_t vres = __riscv_vredsum_vs_i32m8_i32m1(vacc, vzero, vlmax);
      __riscv_vse32_v_i32m1(result_row + c, vres, 1);
    }
  }
}

int main() {
  mcontext0_write_value = 0x01;
  asm volatile("csrw 0x7C0, %0" : : "r"(mcontext0_write_value));
  MatMul(kLhsRows, kInner, kRhsCols, lhs_input, rhs_input, result_output);
  mcontext0_write_value = 0x00;
  asm volatile("csrw 0x7C0, %0" : : "r"(mcontext0_write_value));
  return 0;
}
