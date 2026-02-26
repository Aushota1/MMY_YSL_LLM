# lerning_data — датасет RTL-модулей по классам

Папка для хранения множества файлов с кодом **одного модуля** (SystemVerilog/Verilog). Модули разнесены по классам назначения (RTL-дизайн, синтезируемые модули).

## Формат файлов

- **Один модуль — один файл .txt**
- Содержимое: объявление `module ... endmodule`, как в примере `data/sum_3.txt` (один модуль на файл).
- Имя файла — по желанию (например, по имени модуля: `adder_4bit.txt`, `mux_2to1.txt`).

Пример формата (по аналогии с `data/sum_3.txt`):

```
module adder_4bit (
    input  logic [3:0] a, b,
    input  logic cin,
    output logic [3:0] sum,
    output logic cout
);
    assign {cout, sum} = a + b + cin;
endmodule
```

## Классы (папка → назначение)

| Путь | Класс |
|------|--------|
| **combinational/** | Комбинационные модули (то, что превращается в железо без памяти) |
| `combinational/mux/` | Мультиплексоры |
| `combinational/decoder/` | Дешифраторы |
| `combinational/arithmetic/` | Арифметические блоки: сумматоры, ALU |
| **sequential/** | Последовательностные модули |
| `sequential/registers/` | Регистры |
| `sequential/counters/` | Счётчики |
| `sequential/fsm/` | Конечные автоматы (FSM) |
| **memory/** | Память |
| `memory/ram_rom/` | Модули RAM/ROM |
| `memory/fifo/` | FIFO |
| **interface/** | Интерфейсные модули |
| `interface/uart/` | UART |
| `interface/spi/` | SPI |
| `interface/i2c/` | I2C |
| `interface/ethernet_mac/` | Ethernet MAC |
| **processor/** | Процессорные модули |
| `processor/alu/` | ALU |
| `processor/interrupt_controller/` | Контроллеры прерываний |
| `processor/cache/` | Кэш-память |

В каждой листовой папке размещайте только .txt файлы с кодом одного модуля.
