// Exception control flow: nested try/catch/finally, finally-runs-on-every-path,
// and rethrow. The native translation has to reproduce the exception table edges
// and the finally duplication exactly (Phase 6 lift / Phase 8 verify).
public class Exceptions {
    static int finallyCounter = 0;

    static int nested(int sel) {
        try {
            try {
                if (sel == 0) throw new IllegalStateException("inner");
                if (sel == 1) return 10;
                return 20;
            } finally {
                finallyCounter++;
            }
        } catch (IllegalStateException e) {
            return -1;
        } finally {
            finallyCounter += 100;
        }
    }

    static int rethrow() {
        try {
            throw new RuntimeException("first");
        } catch (RuntimeException e) {
            try {
                throw new IllegalArgumentException("second", e);
            } catch (IllegalArgumentException e2) {
                return e2.getCause().getMessage().length();
            }
        }
    }

    public static void main(String[] args) {
        System.out.println(nested(0));
        System.out.println(nested(1));
        System.out.println(nested(2));
        System.out.println(finallyCounter);
        System.out.println(rethrow());
    }
}
