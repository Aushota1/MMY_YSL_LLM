"""
Генерация по 100 примеров RTL для каждой категории в lerning_data.
Запуск из корня проекта: python Layer_ML/lerning_data/generate_100_per_category.py
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
N_PER = 100


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def gen_mux():
    d = os.path.join(BASE, "combinational", "mux")
    ensure_dir(d)
    for i in range(N_PER):
        w = [2, 4, 8, 16, 32, 64][i % 6]
        n = (i // 6) % 3  # 2:1, 4:1, 8:1
        if n == 0:
            write(os.path.join(d, f"mux_2to1_{i:03d}.txt"), f"""module mux_2to1_{i:03d} #(parameter W = {w}) (
    input  logic [W-1:0] a, b,
    input  logic         sel,
    output logic [W-1:0] y
);
    assign y = sel ? b : a;
endmodule
""")
        elif n == 1:
            write(os.path.join(d, f"mux_4to1_{i:03d}.txt"), f"""module mux_4to1_{i:03d} #(parameter W = {w}) (
    input  logic [W-1:0] a, b, c, d,
    input  logic [1:0]   sel,
    output logic [W-1:0] y
);
    assign y = (sel == 2'b00) ? a : (sel == 2'b01) ? b : (sel == 2'b10) ? c : d;
endmodule
""")
        else:
            write(os.path.join(d, f"mux_8to1_{i:03d}.txt"), f"""module mux_8to1_{i:03d} #(parameter W = {w}) (
    input  logic [W-1:0] d0, d1, d2, d3, d4, d5, d6, d7,
    input  logic [2:0]   sel,
    output logic [W-1:0] y
);
    always_comb case (sel)
        3'b000: y = d0; 3'b001: y = d1; 3'b010: y = d2; 3'b011: y = d3;
        3'b100: y = d4; 3'b101: y = d5; 3'b110: y = d6; 3'b111: y = d7;
        default: y = '0;
    endcase
endmodule
""")


def gen_decoder():
    d = os.path.join(BASE, "combinational", "decoder")
    ensure_dir(d)
    for i in range(N_PER):
        n_in = [2, 3, 4][i % 3]
        n_out = 1 << n_in
        write(os.path.join(d, f"decoder_{i:03d}.txt"), f"""module decoder_{i:03d} (
    input  logic [{n_in-1}:0] a,
    input  logic              en,
    output logic [{n_out-1}:0] y
);
    assign y = en ? ({n_out}'b1 << a) : '0;
endmodule
""")


def gen_arithmetic():
    d = os.path.join(BASE, "combinational", "arithmetic")
    ensure_dir(d)
    for i in range(N_PER):
        w = [4, 8, 16, 32][i % 4]
        t = i % 7
        if t == 0:
            write(os.path.join(d, f"adder_{i:03d}.txt"), f"""module adder_{i:03d} (
    input  logic [{w-1}:0] a, b,
    input  logic cin,
    output logic [{w-1}:0] sum,
    output logic cout
);
    assign {{cout, sum}} = a + b + cin;
endmodule
""")
        elif t == 1:
            write(os.path.join(d, f"sub_{i:03d}.txt"), f"""module sub_{i:03d} (
    input  logic [{w-1}:0] a, b,
    output logic [{w-1}:0] diff,
    output logic borrow
);
    assign {{borrow, diff}} = a - b;
endmodule
""")
        elif t == 2:
            write(os.path.join(d, f"mult_{i:03d}.txt"), f"""module mult_{i:03d} (
    input  logic [{w-1}:0] a, b,
    output logic [{2*w-1}:0] p
);
    assign p = a * b;
endmodule
""")
        elif t == 3:
            write(os.path.join(d, f"and_n_{i:03d}.txt"), f"""module and_n_{i:03d} #(parameter N = {w}) (
    input  logic [N-1:0] a, b,
    output logic [N-1:0] y
);
    assign y = a & b;
endmodule
""")
        elif t == 4:
            write(os.path.join(d, f"or_n_{i:03d}.txt"), f"""module or_n_{i:03d} #(parameter N = {w}) (
    input  logic [N-1:0] a, b,
    output logic [N-1:0] y
);
    assign y = a | b;
endmodule
""")
        elif t == 5:
            write(os.path.join(d, f"xor_n_{i:03d}.txt"), f"""module xor_n_{i:03d} #(parameter N = {w}) (
    input  logic [N-1:0] a, b,
    output logic [N-1:0] y
);
    assign y = a ^ b;
endmodule
""")
        else:
            write(os.path.join(d, f"half_adder_{i:03d}.txt"), """module half_adder (
    input  logic a, b,
    output logic sum, carry
);
    assign sum = a ^ b;
    assign carry = a & b;
endmodule
""")


def gen_registers():
    d = os.path.join(BASE, "sequential", "registers")
    ensure_dir(d)
    for i in range(N_PER):
        n = [1, 2, 4, 8, 16, 32, 64][i % 7]
        write(os.path.join(d, f"reg_{i:03d}.txt"), f"""module reg_{i:03d} #(parameter N = {n}) (
    input  logic clk, rst_n, en,
    input  logic [N-1:0] d,
    output logic [N-1:0] q
);
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) q <= '0;
        else if (en) q <= d;
endmodule
""")


def gen_counters():
    d = os.path.join(BASE, "sequential", "counters")
    ensure_dir(d)
    for i in range(N_PER):
        w = [4, 8, 16, 32][i % 4]
        kind = i % 3
        if kind == 0:
            write(os.path.join(d, f"counter_up_{i:03d}.txt"), f"""module counter_up_{i:03d} #(parameter W = {w}) (
    input  logic clk, rst_n, en,
    output logic [W-1:0] q
);
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) q <= '0;
        else if (en) q <= q + 1;
endmodule
""")
        elif kind == 1:
            max_val = (1 << w) - 1
            write(os.path.join(d, f"counter_down_{i:03d}.txt"), f"""module counter_down_{i:03d} #(parameter W = {w}) (
    input  logic clk, rst_n, en,
    output logic [W-1:0] q
);
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) q <= {w}'d{max_val};
        else if (en) q <= q - 1;
endmodule
""")
        else:
            write(os.path.join(d, f"counter_updown_{i:03d}.txt"), f"""module counter_updown_{i:03d} #(parameter W = {w}) (
    input  logic clk, rst_n, en, up,
    output logic [W-1:0] q
);
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) q <= '0;
        else if (en) q <= up ? q + 1 : q - 1;
endmodule
""")


def gen_fsm():
    d = os.path.join(BASE, "sequential", "fsm")
    ensure_dir(d)
    for i in range(N_PER):
        n_states = 2 + (i % 5)
        write(os.path.join(d, f"fsm_{i:03d}.txt"), f"""module fsm_{i:03d} (
    input  logic clk, rst_n, start, done,
    output logic [2:0] state
);
    localparam S0 = 3'd0, S1 = 3'd1, S2 = 3'd2, S3 = 3'd3, S4 = 3'd4;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S0;
        else case (state)
            S0: state <= start ? S1 : S0;
            S1: state <= S2;
            S2: state <= done ? S3 : S2;
            S3: state <= S0;
            default: state <= S0;
        endcase
endmodule
""")


def gen_ram_rom():
    d = os.path.join(BASE, "memory", "ram_rom")
    ensure_dir(d)
    for i in range(N_PER):
        a = [4, 6, 8, 10][i % 4]
        dw = [8, 16, 32][i % 3]
        write(os.path.join(d, f"sync_ram_{i:03d}.txt"), f"""module sync_ram_{i:03d} #(parameter A = {a}, D = {dw}) (
    input  logic clk, we,
    input  logic [A-1:0] addr,
    input  logic [D-1:0] wdata,
    output logic [D-1:0] rdata
);
    logic [D-1:0] mem [0:(1<<A)-1];
    always_ff @(posedge clk) begin
        if (we) mem[addr] <= wdata;
        rdata <= mem[addr];
    end
endmodule
""")


def gen_fifo():
    d = os.path.join(BASE, "memory", "fifo")
    ensure_dir(d)
    for i in range(N_PER):
        w = [4, 8, 16, 32][i % 4]
        depth = [8, 16, 32, 64][i % 4]
        write(os.path.join(d, f"fifo_{i:03d}.txt"), f"""module fifo_{i:03d} #(parameter W = {w}, D = {depth}) (
    input  logic clk, rst_n, push, pop,
    input  logic [W-1:0] wdata,
    output logic [W-1:0] rdata,
    output logic full, empty
);
    logic [W-1:0] mem [0:D-1];
    logic [$clog2(D):0] wr, rd;
    assign full = (wr[$clog2(D)] != rd[$clog2(D)]) && (wr[$clog2(D)-1:0] == rd[$clog2(D)-1:0]);
    assign empty = (wr == rd);
    assign rdata = mem[rd[$clog2(D)-1:0]];
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) {{wr, rd}} <= '0;
        else begin
            if (push && !full) begin mem[wr[$clog2(D)-1:0]] <= wdata; wr <= wr + 1; end
            if (pop && !empty) rd <= rd + 1;
        end
endmodule
""")


def gen_uart():
    d = os.path.join(BASE, "interface", "uart")
    ensure_dir(d)
    for i in range(N_PER):
        div = [434, 868, 1302][i % 3]
        write(os.path.join(d, f"uart_tx_{i:03d}.txt"), f"""module uart_tx_{i:03d} #(parameter CLK_DIV = {div}) (
    input  logic clk, rst_n, start,
    input  logic [7:0] data,
    output logic tx, busy
);
    logic [9:0] shift;
    logic [3:0] bit_cnt;
    logic [15:0] div;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) {{ tx, busy, div, bit_cnt, shift }} <= '0;
        else if (!busy && start) begin busy <= 1; shift <= {{1'b1, data, 1'b0}}; bit_cnt <= 0; div <= 0; end
        else if (busy)
            if (div == CLK_DIV-1) begin div <= 0; tx <= shift[0]; shift <= {{1'b0, shift[9:1]}}; bit_cnt <= bit_cnt + 1; if (bit_cnt == 9) busy <= 0; end
            else div <= div + 1;
endmodule
""")


def gen_spi():
    d = os.path.join(BASE, "interface", "spi")
    ensure_dir(d)
    for i in range(N_PER):
        w = [8, 16, 32][i % 3]
        write(os.path.join(d, f"spi_master_{i:03d}.txt"), f"""module spi_master_{i:03d} #(parameter W = {w}) (
    input  logic clk, rst_n, start,
    input  logic [W-1:0] mosi_data,
    output logic [W-1:0] miso_data,
    output logic sclk, mosi, cs_n, busy
);
    logic [W-1:0] mosi_sr, miso_sr;
    logic [4:0] cnt;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) {{ busy, cs_n, sclk, cnt }} <= '0;
        else if (!busy && start) begin mosi_sr <= mosi_data; cs_n <= 0; cnt <= 0; busy <= 1; end
        else if (busy) begin sclk <= ~sclk; if (~sclk) begin mosi <= mosi_sr[W-1]; mosi_sr <= {{mosi_sr[W-2:0], 1'b0}}; cnt <= cnt + 1; if (cnt == W-1) begin busy <= 0; cs_n <= 1; miso_data <= miso_sr; end end end
endmodule
""")


def gen_i2c():
    d = os.path.join(BASE, "interface", "i2c")
    ensure_dir(d)
    for i in range(N_PER):
        write(os.path.join(d, f"i2c_stub_{i:03d}.txt"), f"""module i2c_stub_{i:03d} (
    input  logic clk, rst_n, start,
    input  logic [7:0] addr, wdata,
    output logic [7:0] rdata,
    inout  wire  sda, scl,
    output logic busy
);
    assign sda = 1'bz;
    assign scl = 1'bz;
    assign rdata = 8'b0;
    assign busy = 1'b0;
endmodule
""")


def gen_ethernet_mac():
    d = os.path.join(BASE, "interface", "ethernet_mac")
    ensure_dir(d)
    for i in range(N_PER):
        write(os.path.join(d, f"eth_mac_{i:03d}.txt"), f"""module eth_mac_{i:03d} (
    input  logic clk, rst_n,
    input  logic [7:0] tx_data,
    input  logic tx_valid, tx_last,
    output logic tx_ready,
    output logic [7:0] rx_data,
    output logic rx_valid, rx_last
);
    assign tx_ready = 1'b1;
    assign rx_data = 8'b0;
    assign rx_valid = 1'b0;
    assign rx_last = 1'b0;
endmodule
""")


def gen_alu():
    d = os.path.join(BASE, "processor", "alu")
    ensure_dir(d)
    for i in range(N_PER):
        w = [4, 8, 16, 32][i % 4]
        write(os.path.join(d, f"alu_{i:03d}.txt"), f"""module alu_{i:03d} #(parameter W = {w}) (
    input  logic [W-1:0] a, b,
    input  logic [2:0]   op,
    output logic [W-1:0] y,
    output logic zero, carry
);
    logic [W:0] result;
    always_comb case (op)
        3'b000: result = a + b;
        3'b001: result = a - b;
        3'b010: result = a & b;
        3'b011: result = a | b;
        3'b100: result = a ^ b;
        default: result = a;
    endcase
    assign y = result[W-1:0];
    assign carry = result[W];
    assign zero = (y == 0);
endmodule
""")


def gen_interrupt_controller():
    d = os.path.join(BASE, "processor", "interrupt_controller")
    ensure_dir(d)
    for i in range(N_PER):
        n = [4, 8, 16, 32][i % 4]
        write(os.path.join(d, f"irq_ctrl_{i:03d}.txt"), f"""module irq_ctrl_{i:03d} #(parameter N = {n}) (
    input  logic clk, rst_n,
    input  logic [N-1:0] irq, mask, ack,
    output logic [N-1:0] pending,
    output logic         any_irq
);
    logic [N-1:0] latched;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) latched <= '0;
        else latched <= (latched | (irq & mask)) & ~ack;
    assign pending = latched & mask;
    assign any_irq = |pending;
endmodule
""")


def gen_cache():
    d = os.path.join(BASE, "processor", "cache")
    ensure_dir(d)
    for i in range(N_PER):
        line_w = [4, 6, 8][i % 3]
        tag_w = [6, 8, 10][i % 3]
        dw = [32, 64][i % 2]
        write(os.path.join(d, f"cache_{i:03d}.txt"), f"""module cache_{i:03d} #(parameter LINE_W = {line_w}, TAG_W = {tag_w}, D = {dw}) (
    input  logic clk, rst_n, we, re,
    input  logic [TAG_W+LINE_W-1:0] addr,
    input  logic [D-1:0] wdata,
    output logic [D-1:0] rdata,
    output logic hit
);
    localparam LINES = 1 << LINE_W;
    logic [TAG_W-1:0] tag_ram [0:LINES-1];
    logic [D-1:0]     data_ram [0:LINES-1];
    logic [TAG_W-1:0] addr_tag;
    logic [LINE_W-1:0] addr_line;
    assign addr_tag = addr[TAG_W+LINE_W-1:LINE_W];
    assign addr_line = addr[LINE_W-1:0];
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) for (int i = 0; i < LINES; i++) tag_ram[i] <= '0;
        else if (we) begin data_ram[addr_line] <= wdata; tag_ram[addr_line] <= addr_tag; end
    assign rdata = data_ram[addr_line];
    assign hit = (tag_ram[addr_line] == addr_tag);
endmodule
""")


def main():
    gen_mux()
    gen_decoder()
    gen_arithmetic()
    gen_registers()
    gen_counters()
    gen_fsm()
    gen_ram_rom()
    gen_fifo()
    gen_uart()
    gen_spi()
    gen_i2c()
    gen_ethernet_mac()
    gen_alu()
    gen_interrupt_controller()
    gen_cache()
    print("Done. 100 files per category.")


if __name__ == "__main__":
    main()
