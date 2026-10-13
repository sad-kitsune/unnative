// invokedynamic: lambda metafactory (two lambdas, one capturing) and string
// concatenation (makeConcatWithConstants). JNIC emulates invokedynamic in native
// code, so these methods exercise that path.
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;

public class Indy {
    public static void main(String[] args) {
        IntBinaryOperator add = (a, b) -> a + b;           // non-capturing lambda
        int base = 100;
        IntUnaryOperator addBase = x -> x + base;          // captures `base`

        System.out.println(add.applyAsInt(3, 4));
        System.out.println(addBase.applyAsInt(5));

        // String concat via invokedynamic (makeConcatWithConstants).
        int n = 42;
        String s = "n=" + n + ", half=" + (n / 2) + ", flag=" + (n > 0);
        System.out.println(s);
        System.out.println("" + 'c' + 1L + 2.5 + true);
    }
}
