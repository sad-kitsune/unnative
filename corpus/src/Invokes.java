// Every invoke* kind plus field access, object creation and `super`:
//   invokestatic, invokevirtual, invokespecial (<init> and super.m()),
//   invokeinterface (incl. a default method), getfield/putfield,
//   getstatic/putstatic. These map one-to-one onto the JNI idioms the lifter sees.
public class Invokes {
    interface Greeter {
        String greet(String who);
        default String shout(String who) { return greet(who).toUpperCase(); }  // default
    }

    static class Base {
        static int created = 0;                 // putstatic / getstatic
        int id;                                 // getfield / putfield
        Base() { id = ++created; }              // invokespecial <init>
        int describe() { return id; }           // invokevirtual target
    }

    static class Derived extends Base implements Greeter {
        Derived() { super(); }                  // invokespecial super.<init>
        @Override int describe() { return 1000 + super.describe(); }  // invokespecial super.m
        @Override public String greet(String who) { return "hi " + who; }
    }

    static int twice(int x) { return x + x; }   // invokestatic

    public static void main(String[] args) {
        System.out.println(twice(21));
        Derived d = new Derived();
        System.out.println(d.describe());       // invokevirtual -> super call
        Greeter g = d;
        System.out.println(g.greet("world"));   // invokeinterface
        System.out.println(g.shout("world"));   // default method
        System.out.println(Base.created);       // getstatic
    }
}
