// Arithmetic edge cases a faithful rebuild must preserve exactly. One to watch:
// JNIC's native translation does not throw on integer division by zero, so the
// idiv/irem-by-zero handling here is a prime differential-verification target.
public class ArithEdges {
    public static void main(String[] args) {
        // int overflow wraps
        System.out.println(Integer.MAX_VALUE + 1);
        System.out.println(Integer.MIN_VALUE - 1);
        System.out.println(Integer.MIN_VALUE / -1);   // overflow: stays MIN_VALUE

        // long compares
        System.out.println(Long.compare(Long.MAX_VALUE, Long.MIN_VALUE));

        // shifts: distance is masked (& 31 for int, & 63 for long)
        System.out.println(1 << 33);                   // == 1 << 1 == 2
        System.out.println(-8 >> 1);                   // arithmetic
        System.out.println(-8 >>> 28);                 // logical

        // idiv / irem by zero — throws on a normal JVM; the quirk to watch for.
        try {
            int z = 0;
            System.out.println(7 / z);
        } catch (ArithmeticException e) {
            System.out.println("idiv0:" + e.getMessage());
        }
        try {
            int z = 0;
            System.out.println(7 % z);
        } catch (ArithmeticException e) {
            System.out.println("irem0:" + e.getMessage());
        }

        // floating point: NaN is never equal to itself; division by zero is Inf.
        double nan = 0.0 / 0.0;
        System.out.println(nan == nan);
        System.out.println(1.0 / 0.0);
        System.out.println(-1.0 / 0.0);
        System.out.println((int) nan);                 // d2i of NaN is 0
        System.out.println((int) Double.POSITIVE_INFINITY);  // saturates to MAX
    }
}
