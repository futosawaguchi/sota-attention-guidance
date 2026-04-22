package jp.vstone.sotatest;

import java.awt.Color;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.DatagramChannel;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import jp.vstone.RobotLib.CRobotMem;
import jp.vstone.RobotLib.CRobotPose;
import jp.vstone.RobotLib.CRobotUtil;
import jp.vstone.RobotLib.CSotaMotion;

public class SotaController {

    static final int RECEIVE_PORT = 9980;

    static final String[] PARTS = {
        "Waist_Y", "RShoulder_P", "RElbow_P",
        "LShoulder_P", "LElbow_P", "Head_Y", "Head_P", "Head_R"
    };

    static final short[][] LIMIT_VALUE = {
        {-1200, 1200},
        {-1400,  800},
        { -900,  650},
        { -800, 1400},
        { -650,  900},
        {-1400, 1400},
        { -290,  110},
        { -300,  350}
    };

    static short[] vals = {0, -900, 0, 900, 0, 0, 0, 0};
    static volatile boolean isBusy       = false;
    static volatile String  pendingLED    = null;
    static volatile String  pendingMotion = null;
    static volatile boolean servoUpdated  = false;

    static final Object servoLock  = new Object();
    static final Object motionLock = new Object();

    static CRobotMem   mem;
    static CSotaMotion motion;
    static CRobotPose  servoPose;
    static Byte[]      ids = {1, 2, 3, 4, 5, 6, 7, 8};

    public static void main(String[] args) throws Exception {
        mem    = new CRobotMem();
        motion = new CSotaMotion(mem);

        if (!mem.Connect()) {
            System.out.println("ERROR: mem.Connect() failed");
            return;
        }

        motion.InitRobot_Sota();
        motion.ServoOn();

        // サーボ用pose（LED情報なし）
        servoPose = new CRobotPose();
        servoPose.SetPose(
            new Byte[]{1, 2, 3, 4, 5, 6, 7, 8},
            new Short[]{0, -900, 0, 900, 0, 0, 0, 0}
        );
        motion.play(servoPose, 500);
        CRobotUtil.wait(200);

        // 初期LEDは別poseで送る
        CRobotPose initLED = new CRobotPose();
        initLED.setLED_Sota(Color.GREEN, Color.GREEN, 128, Color.GREEN);
        motion.play(initLED, 200);
        CRobotUtil.wait(600);

        System.out.println("SotaController ready. UDP port: " + RECEIVE_PORT);

        // UDP受信スレッド
        Thread udpThread = new Thread(() -> udpLoop());
        udpThread.setDaemon(true);
        udpThread.start();

        // メインループ：motion.play()はここだけから呼ぶ
        while (true) {
            String led       = null;
            String motionCmd = null;

            synchronized (motionLock) {
                led       = pendingLED;
                motionCmd = pendingMotion;
                pendingLED    = null;
                pendingMotion = null;
            }

            if (motionCmd != null) {
                isBusy = true;
                try {
                    doMotion(motionCmd);
                } finally {
                    isBusy = false;
                }

            } else if (led != null) {
                setLED(led);

            } else if (servoUpdated) {
                servoUpdated = false;
                synchronized (servoLock) {
                    Short[] shortVals = new Short[vals.length];
                    for (int i = 0; i < vals.length; i++) shortVals[i] = vals[i];
                    servoPose.SetPose(ids, shortVals);
                }
                motion.play(servoPose, 50);
                CRobotUtil.wait(50);

            } else {
                CRobotUtil.wait(10);
            }
        }
    }

    // ========== UDP受信ループ ==========
    static void udpLoop() {
        try {
            DatagramChannel channel = DatagramChannel.open();
            channel.socket().bind(new InetSocketAddress(RECEIVE_PORT));
            ByteBuffer buf = ByteBuffer.allocate(50000);
            System.out.println("UDP listening...");

            while (true) {
                ((Buffer) buf).clear();
                channel.receive(buf);
                ((Buffer) buf).flip();
                byte[] data = new byte[buf.limit()];
                buf.get(data);
                String json = new String(data);
                System.out.println("Received: " + json);

                // LED・モーション
                String led = jsonGet(json, "LED");
                if (led != null) {
                    synchronized (motionLock) { pendingLED = led; }
                }

                String mot = jsonGet(json, "Motion");
                if (mot != null) {
                    synchronized (motionLock) { pendingMotion = mot; }
                }

                // モーション中はサーボ値更新をスキップ
                if (isBusy) continue;

                // サーボ値を更新するだけ
                synchronized (servoLock) {
                    for (int i = 0; i < PARTS.length; i++) {
                        String tmp = jsonGet(json, PARTS[i]);
                        if (tmp != null) {
                            try {
                                short val = Short.parseShort(tmp);
                                if      (val < LIMIT_VALUE[i][0]) val = LIMIT_VALUE[i][0];
                                else if (val > LIMIT_VALUE[i][1]) val = LIMIT_VALUE[i][1];
                                vals[i] = val;
                                servoUpdated = true;
                            } catch (NumberFormatException e) {}
                        }
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // ========== JSON解析（正規表現・軽量） ==========
    static String jsonGet(String json, String key) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*\"?(-?[\\w.]+)\"?");
        Matcher m = p.matcher(json);
        if (m.find()) {
            return m.group(1);
        }
        return null;
    }

    // ========== LED制御 ==========
    static void setLED(String color) throws Exception {
        CRobotPose p = new CRobotPose();
        switch (color) {
            case "blue":  p.setLED_Sota(Color.BLUE,  Color.BLUE,  255, Color.BLUE);  break;
            case "white": p.setLED_Sota(Color.WHITE, Color.WHITE, 255, Color.WHITE); break;
            case "green": p.setLED_Sota(Color.GREEN, Color.GREEN, 128, Color.GREEN); break;
            case "red":   p.setLED_Sota(Color.RED,   Color.RED,   255, Color.RED);   break;
            case "off":   p.setLED_Sota(Color.BLACK, Color.BLACK, 0,   Color.BLACK); break;
            default: return;
        }
        motion.play(p, 80);
        CRobotUtil.wait(80);
    }

    // ========== モーション制御 ==========
    static void doMotion(String cmd) throws Exception {
        CRobotPose p = new CRobotPose();
        switch (cmd) {
            case "right_hand_up":
                p.SetPose(new Byte[]{CSotaMotion.SV_R_SHOULDER, CSotaMotion.SV_R_ELBOW},
                          new Short[]{-500, -300});
                motion.play(p, 500); CRobotUtil.wait(1500);
                p = new CRobotPose();
                p.SetPose(new Byte[]{CSotaMotion.SV_R_SHOULDER, CSotaMotion.SV_R_ELBOW},
                          new Short[]{-900, 0});
                motion.play(p, 500); CRobotUtil.wait(500);
                break;

            case "left_hand_up":
                p.SetPose(new Byte[]{CSotaMotion.SV_L_SHOULDER, CSotaMotion.SV_L_ELBOW},
                          new Short[]{500, 300});
                motion.play(p, 500); CRobotUtil.wait(1500);
                p = new CRobotPose();
                p.SetPose(new Byte[]{CSotaMotion.SV_L_SHOULDER, CSotaMotion.SV_L_ELBOW},
                          new Short[]{900, 0});
                motion.play(p, 500); CRobotUtil.wait(500);
                break;

            case "both_hands_up":
                p.SetPose(
                    new Byte[]{CSotaMotion.SV_L_SHOULDER, CSotaMotion.SV_L_ELBOW,
                               CSotaMotion.SV_R_SHOULDER, CSotaMotion.SV_R_ELBOW},
                    new Short[]{500, 300, -500, -300});
                motion.play(p, 500); CRobotUtil.wait(1500);
                p = new CRobotPose();
                p.SetPose(
                    new Byte[]{CSotaMotion.SV_L_SHOULDER, CSotaMotion.SV_L_ELBOW,
                               CSotaMotion.SV_R_SHOULDER, CSotaMotion.SV_R_ELBOW},
                    new Short[]{900, 0, -900, 0});
                motion.play(p, 500); CRobotUtil.wait(500);
                break;

            case "bye_bye":
                p.SetPose(new Byte[]{CSotaMotion.SV_R_SHOULDER, CSotaMotion.SV_R_ELBOW},
                          new Short[]{-500, -300});
                motion.play(p, 400); CRobotUtil.wait(400);
                for (int i = 0; i < 3; i++) {
                    p = new CRobotPose();
                    p.SetPose(new Byte[]{CSotaMotion.SV_R_ELBOW}, new Short[]{-500});
                    motion.play(p, 250); CRobotUtil.wait(250);
                    p = new CRobotPose();
                    p.SetPose(new Byte[]{CSotaMotion.SV_R_ELBOW}, new Short[]{-100});
                    motion.play(p, 250); CRobotUtil.wait(250);
                }
                p = new CRobotPose();
                p.SetPose(new Byte[]{CSotaMotion.SV_R_SHOULDER, CSotaMotion.SV_R_ELBOW},
                          new Short[]{-900, 0});
                motion.play(p, 500); CRobotUtil.wait(500);
                break;

            case "nod":
                p.SetPose(new Byte[]{CSotaMotion.SV_HEAD_P}, new Short[]{100});
                motion.play(p, 300); CRobotUtil.wait(400);
                p = new CRobotPose();
                p.SetPose(new Byte[]{CSotaMotion.SV_HEAD_P}, new Short[]{-100});
                motion.play(p, 300); CRobotUtil.wait(400);
                p = new CRobotPose();
                p.SetPose(new Byte[]{CSotaMotion.SV_HEAD_P}, new Short[]{0});
                motion.play(p, 300); CRobotUtil.wait(300);
                break;

            case "shake_head":
                p.SetPose(new Byte[]{CSotaMotion.SV_HEAD_Y}, new Short[]{400});
                motion.play(p, 300); CRobotUtil.wait(400);
                p = new CRobotPose();
                p.SetPose(new Byte[]{CSotaMotion.SV_HEAD_Y}, new Short[]{-400});
                motion.play(p, 300); CRobotUtil.wait(400);
                p = new CRobotPose();
                p.SetPose(new Byte[]{CSotaMotion.SV_HEAD_Y}, new Short[]{0});
                motion.play(p, 300); CRobotUtil.wait(300);
                break;

            default:
                System.out.println("Unknown motion: " + cmd);
                return;
        }

        // モーション後に現在のサーボ値で姿勢を復元
        synchronized (servoLock) {
            Short[] shortVals = new Short[vals.length];
            for (int i = 0; i < vals.length; i++) shortVals[i] = vals[i];
            servoPose.SetPose(ids, shortVals);
        }
        motion.play(servoPose, 300);
        CRobotUtil.wait(300);
    }
}