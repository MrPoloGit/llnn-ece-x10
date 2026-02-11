// ============================================================================
// uart_wrapper.sv — UART-based I/O wrapper for LUTNN on iCEBreaker
//
// Protocol:
//   1. Host sends N_BYTES bytes over UART (MSB first, packed into input vector)
//   2. Wrapper feeds assembled input to the combinational LUTNN
//   3. Wrapper sends 1 byte back: the predicted class index
//
// Parameters are pulled from Globals.sv via `include.
// ============================================================================

`include "Globals.sv"

module uart_wrapper (
    input  wire clk,      // 12 MHz oscillator
    input  wire uart_rx,  // FTDI TX -> FPGA RX
    output wire uart_tx   // FPGA TX -> FTDI RX
);

  // ---- Parameters ----
  localparam PRESCALE = 13;  // 12_000_000 / (115200 * 8) ≈ 13
  localparam N_BYTES = (NET_INPUTS + 7) / 8;  // ceil(NET_INPUTS / 8)
  localparam PADDED_BITS = N_BYTES * 8;

  // ---- Reset generator (simple power-on reset) ----
  reg [3:0] rst_cnt = 4'hF;
  wire rst = rst_cnt[3];
  always @(posedge clk) begin
    if (rst_cnt != 0) rst_cnt <= rst_cnt - 1;
  end

  // ---- UART RX ----
  wire [7:0] rx_data;
  wire       rx_valid;
  reg        rx_ready;

  uart_rx u_rx (
      .clk          (clk),
      .rst          (rst),
      .m_axis_tdata (rx_data),
      .m_axis_tvalid(rx_valid),
      .m_axis_tready(rx_ready),
      .rxd          (uart_rx),
      .busy         (),
      .overrun_error(),
      .frame_error  (),
      .prescale     (PRESCALE)
  );

  // ---- UART TX ----
  wire       tx_ready;
  reg  [7:0] tx_data;
  reg        tx_valid;

  uart_tx u_tx (
      .clk          (clk),
      .rst          (rst),
      .s_axis_tdata (tx_data),
      .s_axis_tvalid(tx_valid),
      .s_axis_tready(tx_ready),
      .txd          (uart_tx),
      .busy         (),
      .prescale     (PRESCALE)
  );

  // ---- Input shift register ----
  reg [PADDED_BITS-1:0] input_shift;
  reg [$clog2(N_BYTES):0] byte_cnt;

  // ---- LUTNN instance (combinational) ----
  wire [NET_INPUTS-1:0] net_input = input_shift[NET_INPUTS-1:0];
  wire [NET_OUTPUT_BITS-1:0] net_output;

  top lutnn (
      .NET_I(net_input),
      .NET_O(net_output)
  );

  // ---- FSM ----
  localparam S_IDLE = 2'd0;
  localparam S_RECV = 2'd1;
  localparam S_SEND = 2'd2;

  reg [1:0] state;

  always @(posedge clk) begin
    if (rst) begin
      state    <= S_IDLE;
      byte_cnt <= 0;
      rx_ready <= 1;
      tx_valid <= 0;
      tx_data  <= 0;
      input_shift <= 0;
    end else begin
      // Default: deassert tx_valid after handshake
      if (tx_valid && tx_ready) tx_valid <= 0;

      case (state)
        S_IDLE: begin
          byte_cnt <= 0;
          rx_ready <= 1;
          state    <= S_RECV;
        end

        S_RECV: begin
          rx_ready <= 1;
          if (rx_valid && rx_ready) begin
            // Shift in MSB-first: new byte goes to the top
            input_shift <= {input_shift[PADDED_BITS-9:0], rx_data};
            byte_cnt    <= byte_cnt + 1;

            if (byte_cnt == N_BYTES - 1) begin
              // All bytes received — send result
              rx_ready <= 0;
              state    <= S_SEND;
            end
          end
        end

        S_SEND: begin
          rx_ready <= 0;
          if (!tx_valid || tx_ready) begin
            // Send the classification result as a single byte
            tx_data  <= {{(8-NET_OUTPUT_BITS){1'b0}}, net_output};
            tx_valid <= 1;
            state    <= S_IDLE;
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
