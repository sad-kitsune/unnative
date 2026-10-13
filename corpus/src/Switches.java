// Every switch shape: tableswitch (dense), lookupswitch (sparse), String switch
// (hashCode + equals, two-level), and enum switch (the synthetic $SwitchMap array).
// These stress the CFG recovery (Phase 6 cfg.py) hardest.
public class Switches {
    enum Color { RED, GREEN, BLUE }

    static String dense(int x) {
        switch (x) {
            case 0: return "zero";
            case 1: return "one";
            case 2: return "two";
            case 3: return "three";
            default: return "many";
        }
    }

    static String sparse(int x) {
        switch (x) {
            case 1: return "a";
            case 100: return "b";
            case 10000: return "c";
            default: return "d";
        }
    }

    static String str(String s) {
        switch (s) {
            case "alpha": return "1";
            case "beta": return "2";
            case "gamma": return "3";
            default: return "0";
        }
    }

    static String color(Color c) {
        switch (c) {
            case RED: return "r";
            case GREEN: return "g";
            case BLUE: return "b";
            default: return "?";
        }
    }

    public static void main(String[] args) {
        for (int i = 0; i <= 4; i++) System.out.print(dense(i));
        System.out.println();
        System.out.println(sparse(1) + sparse(100) + sparse(10000) + sparse(7));
        System.out.println(str("alpha") + str("beta") + str("gamma") + str("x"));
        System.out.println(color(Color.RED) + color(Color.GREEN) + color(Color.BLUE));
    }
}
